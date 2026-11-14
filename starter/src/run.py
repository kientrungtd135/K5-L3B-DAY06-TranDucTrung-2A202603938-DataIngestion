"""Run the fixed A/B/C experiment on the selected benchmark split."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import psycopg
from dotenv import dotenv_values

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_markdown
from .models import Chunk, DocumentPage
from .store import PgVectorStore


_STARTER_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _STARTER_DIR.parent
_RESULTS_DIR = _STARTER_DIR / "results"
_STRATEGIES = ("A", "B", "C")
_PRICE_URL = "https://ckey.vn/en/llm-api/m/text-embedding-3-small"
_INPUT_PRICE_PER_MILLION = 0.0448
_MIN_PRICE_PER_REQUEST = 0.00024


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _preflight() -> dict[str, Any]:
    manifest_path = _STARTER_DIR / "data" / "manifest.json"
    benchmark_path = _REPO_ROOT / "benchmark" / "questions.json"
    manifest_bytes = manifest_path.read_bytes()
    benchmark_bytes = benchmark_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    benchmark = json.loads(benchmark_bytes.decode("utf-8-sig"))

    sources = manifest.get("sources")
    _require(isinstance(sources, list), "Manifest must contain a sources list")
    corpus_root = (_STARTER_DIR / "data" / "corpus").resolve()
    documents: dict[str, DocumentPage] = {}
    source_by_id: dict[str, dict[str, Any]] = {}
    paths: set[Path] = set()
    body_hashes: set[str] = set()
    category_counts: Counter[str] = Counter()
    unit_ids: set[str] = set()
    content_hasher = hashlib.sha256()
    total_units = 0

    for source in sources:
        source_id = source.get("source_id")
        _require(isinstance(source_id, str) and source_id, "Manifest source_id is missing")
        _require(source_id not in source_by_id, f"Duplicate manifest source_id: {source_id}")

        relative_path = source.get("local_path")
        _require(isinstance(relative_path, str) and relative_path, f"Missing local_path for {source_id}")
        source_path = (_STARTER_DIR / relative_path).resolve()
        try:
            source_path.relative_to(corpus_root)
        except ValueError as exc:
            raise ValueError(f"Manifest path is outside the corpus: {relative_path}") from exc
        _require(source_path not in paths, f"Duplicate manifest file path: {relative_path}")
        paths.add(source_path)
        _require(source_path.is_file(), f"Manifest-listed article is missing: {relative_path}")

        document = load_markdown(source_path)
        source_metadata = source.get("metadata")
        _require(isinstance(source_metadata, dict), f"Missing metadata for {source_id}")
        _require(document.document_id == source_id, f"Document ID mismatch: {source_id}")
        _require(
            document.source == source_metadata.get("url") == source.get("source_path_or_url"),
            f"Source URL mismatch: {source_id}",
        )
        _require(
            document.metadata.get("category") == source_metadata.get("category"),
            f"Category mismatch: {source_id}",
        )
        _require(
            document.metadata.get("title") == source.get("title"),
            f"Title mismatch: {source_id}",
        )
        _require(bool(document.text.strip()), f"Article body is empty: {source_id}")
        _require(
            document.metadata.get("char_count") == len(document.text),
            f"Loaded character count is inconsistent: {source_id}",
        )

        units = source.get("units")
        _require(isinstance(units, list) and len(units) == 1, f"Expected one Markdown article unit: {source_id}")
        unit = units[0]
        unit_id = unit.get("unit_id")
        _require(isinstance(unit_id, str) and unit_id, f"Missing unit_id: {source_id}")
        _require(unit_id not in unit_ids, f"Duplicate manifest unit_id: {unit_id}")
        unit_ids.add(unit_id)
        _require(
            unit_id == document.metadata.get("unit_id")
            and unit.get("position") == 1
            and unit.get("start_line") == document.metadata.get("start_line")
            and unit.get("end_line") == document.metadata.get("end_line")
            and unit.get("char_count") == len(document.text)
            and document.page == 1,
            f"Manifest unit boundary/character count mismatch: {source_id}",
        )

        normalized_body = _normalized(document.text)
        body_hash = hashlib.sha256(normalized_body.encode("utf-8")).hexdigest()
        _require(body_hash not in body_hashes, f"Duplicate normalized article body: {source_id}")
        body_hashes.add(body_hash)
        content_hasher.update(source_id.encode("utf-8"))
        content_hasher.update(b"\0")
        content_hasher.update(document.text.encode("utf-8"))
        content_hasher.update(b"\0")

        category = document.metadata.get("category")
        _require(isinstance(category, str) and category, f"Missing category: {source_id}")
        category_counts[category] += 1
        documents[source_id] = document
        source_by_id[source_id] = source
        total_units += len(units)

    _require(len(sources) >= 30, "Manifest contains fewer than 30 source articles")
    _require(
        manifest.get("source_count") == len(sources)
        and manifest.get("unit_count") == total_units
        and total_units >= 30,
        "Manifest source/unit totals do not match its unique listed articles",
    )
    declared_categories = manifest.get("category_counts")
    if declared_categories is not None:
        _require(
            dict(category_counts) == declared_categories,
            "Manifest category totals do not match loaded articles",
        )

    questions = benchmark.get("questions")
    _require(isinstance(questions, list), "Benchmark must contain a questions list")
    question_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    for question in questions:
        question_id = question.get("id")
        _require(
            isinstance(question_id, str) and question_id and question_id not in question_ids,
            f"Missing or duplicate benchmark question ID: {question_id!r}",
        )
        question_ids.add(question_id)
        split = question.get("split")
        _require(split in {"dev", "test"}, f"Invalid benchmark split: {question_id}")
        split_counts[split] += 1
        for field in (
            "question",
            "expected_answer",
            "gold_source_id",
            "gold_page",
            "gold_unit_id",
            "evidence_quote",
        ):
            _require(bool(question.get(field)), f"Missing {field} in benchmark question {question_id}")

        gold_id = question["gold_source_id"]
        _require(gold_id in documents, f"Unknown gold source for {question_id}: {gold_id}")
        gold_document = documents[gold_id]
        gold_source = source_by_id[gold_id]
        gold_units = {unit["unit_id"] for unit in gold_source["units"]}
        _require(
            question["gold_page"] == gold_document.page
            and question["gold_unit_id"] in gold_units,
            f"Gold page/unit mismatch for {question_id}",
        )
        _require(
            _normalized(question["evidence_quote"]) in _normalized(gold_document.text),
            f"Evidence quote is absent from its source for {question_id}",
        )

        requested_filter = question.get("filter") or {}
        _require(isinstance(requested_filter, dict), f"Invalid filter for {question_id}")
        _require(
            set(requested_filter).issubset({"category"}),
            f"Unsupported benchmark filter field for {question_id}",
        )
        if "category" in requested_filter:
            _require(
                requested_filter["category"] == gold_document.metadata.get("category"),
                f"Gold source category conflicts with filter for {question_id}",
            )

    _require(len(questions) == 20, "Benchmark must contain exactly 20 questions")
    _require(
        split_counts == Counter({"dev": 15, "test": 5}),
        f"Expected 15 DEV and 5 TEST questions, found {dict(split_counts)}",
    )
    _require(
        isinstance(benchmark.get("selection_rule"), str)
        and benchmark["selection_rule"],
        "Benchmark selection_rule is missing",
    )

    return {
        "manifest": manifest,
        "benchmark": benchmark,
        "documents": [documents[key] for key in sorted(documents)],
        "source_by_id": source_by_id,
        "questions": questions,
        "split_counts": dict(split_counts),
        "category_counts": dict(sorted(category_counts.items())),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "benchmark_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
        "corpus_content_sha256": content_hasher.hexdigest(),
        "preflight_seconds": None,
    }


def _coherence_sample(documents: list[DocumentPage]) -> dict[str, Any]:
    by_category: dict[str, list[DocumentPage]] = {}
    for document in documents:
        by_category.setdefault(document.metadata["category"], []).append(document)

    items = []
    for category, category_documents in sorted(by_category.items()):
        selected = sorted(category_documents, key=lambda doc: doc.document_id)[:5]
        _require(len(selected) == 5, f"Fewer than five articles in category {category}")
        for document in selected:
            items.append(
                {
                    "category": category,
                    "document_id": document.document_id,
                    "unit_id": document.metadata["unit_id"],
                    "page": document.page,
                    "character_start": 0,
                    "character_end_exclusive": min(500, len(document.text)),
                }
            )
    _require(len(items) == 10, "Coherence sample must contain ten source positions")
    return {
        "selection_rule": "First five document IDs in lexicographic order per category.",
        "position_rule": "Normalized article body character offsets [0, min(500, body length)).",
        "same_positions_for_all_strategies": True,
        "scores_assigned": False,
        "items": items,
    }


def _chunk_statistics(
    chunks: list[Chunk], documents: list[DocumentPage]
) -> dict[str, Any]:
    lengths = sorted(len(chunk.text) for chunk in chunks)
    if lengths:
        p95 = lengths[math.ceil(0.95 * len(lengths)) - 1]
        distribution = {
            "min": lengths[0],
            "median": statistics.median(lengths),
            "p95": p95,
            "max": lengths[-1],
            "mean": statistics.mean(lengths),
        }
    else:
        distribution = None
    return {
        "chunk_count": len(chunks),
        "source_content_characters": sum(len(document.text) for document in documents),
        "chunk_text_characters": sum(lengths),
        "chunk_length_characters": distribution,
    }


def _snapshot_usage(embedder: MyEmbedder) -> tuple[int, int]:
    return embedder.api_call_count, len(embedder.usage_by_request)


def _usage_report(
    embedder: MyEmbedder, snapshot: tuple[int, int]
) -> dict[str, Any]:
    start_calls, start_records = snapshot
    calls = embedder.api_call_count - start_calls
    records = embedder.usage_by_request[start_records:]
    complete = len(records) == calls and all(record is not None for record in records)
    if not complete:
        return {
            "sdk_request_calls": calls,
            "usage_available": False,
            "prompt_tokens": None,
            "total_tokens": None,
            "estimated_cost_usd": None,
            "cost_unknown_reason": "One or more API responses did not include complete usage.",
        }

    valid_records = [record for record in records if record is not None]
    prompt_tokens = sum(record["prompt_tokens"] for record in valid_records)
    total_tokens = sum(record["total_tokens"] for record in valid_records)
    estimated_cost = sum(
        max(
            record["prompt_tokens"] * _INPUT_PRICE_PER_MILLION / 1_000_000,
            _MIN_PRICE_PER_REQUEST,
        )
        for record in valid_records
    )
    return {
        "sdk_request_calls": calls,
        "usage_available": True,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 8),
        "cost_estimation": "Sum per response of max(prompt_tokens * $0.0448/1M, $0.00024 minimum/request).",
    }


def _database_info(database_url: str) -> dict[str, Any]:
    with psycopg.connect(database_url) as connection:
        row = connection.execute(
            """
            SELECT current_setting('server_version'),
                   (SELECT extversion FROM pg_extension WHERE extname = 'vector')
            """
        ).fetchone()
    return {"postgresql": row[0], "pgvector": row[1]}


def _remove_obsolete_strategy_rows(database_url: str, document_ids: list[str]) -> int:
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM public.day6_chunks
                WHERE strategy = ANY(%s)
                  AND NOT (document_id = ANY(%s))
                """,
                (list(_STRATEGIES), document_ids),
            )
            return cursor.rowcount


def _strategy_row_counts(database_url: str) -> dict[str, int]:
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(
            """
            SELECT strategy, count(*)
            FROM public.day6_chunks
            WHERE strategy = ANY(%s)
            GROUP BY strategy
            """,
            (list(_STRATEGIES),),
        ).fetchall()
    return {strategy: count for strategy, count in rows}


def _foreign_strategy_documents(
    database_url: str, document_ids: list[str]
) -> list[tuple[str, str]]:
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT strategy, document_id
            FROM public.day6_chunks
            WHERE strategy = ANY(%s)
              AND NOT (document_id = ANY(%s))
            """,
            (list(_STRATEGIES), document_ids),
        ).fetchall()
    return rows


def _safe_endpoint(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.hostname:
        return "[configured endpoint]"
    host = parsed.hostname
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))


def _package_versions() -> dict[str, str]:
    return {
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "openai": importlib.metadata.version("openai"),
        "psycopg": importlib.metadata.version("psycopg"),
        "pypdf": importlib.metadata.version("pypdf"),
        "python-dotenv": importlib.metadata.version("python-dotenv"),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _select_strategy(
    dev_summary: dict[str, Any], chunk_counts: dict[str, int]
) -> tuple[str, dict[str, Any]]:
    order = {strategy: index for index, strategy in enumerate(_STRATEGIES)}
    ranking = sorted(
        _STRATEGIES,
        key=lambda strategy: (
            -dev_summary[strategy]["dev"]["hit@3"]["hits"],
            -dev_summary[strategy]["dev"]["hit@1"]["hits"],
            chunk_counts[strategy],
            order[strategy],
        ),
    )
    selected = ranking[0]
    winner = dev_summary[selected]["dev"]
    tied = [
        strategy
        for strategy in ranking
        if (
            dev_summary[strategy]["dev"]["hit@3"]["hits"],
            dev_summary[strategy]["dev"]["hit@1"]["hits"],
            chunk_counts[strategy],
        )
        == (
            winner["hit@3"]["hits"],
            winner["hit@1"]["hits"],
            chunk_counts[selected],
        )
    ]
    tie_breaker = (
        f"Tie among {', '.join(tied)} on all metric criteria; "
        f"fixed order {', '.join(_STRATEGIES)} selected {selected}."
        if len(tied) > 1
        else "Resolved by Evidence Hit@3, then Hit@1, then stored chunk count."
    )
    ranking_details = [
        {
            "strategy": strategy,
            "evidence_hit_at_3": dev_summary[strategy]["dev"]["hit@3"],
            "hit_at_1": dev_summary[strategy]["dev"]["hit@1"],
            "stored_chunks": chunk_counts[strategy],
        }
        for strategy in ranking
    ]
    return selected, {
        "selection_rule": "DEV: Evidence Hit@3 descending, Hit@1 descending, stored chunk count ascending.",
        "tie_break_order": list(_STRATEGIES),
        "tie_breaker_applied": tie_breaker,
        "selected_strategy": selected,
        "ranking": ranking_details,
        "test_used_for_selection": False,
    }


def _update_experiment_log(
    summary: dict[str, Any],
    chunk_counts: dict[str, int],
    dimensions: int,
    selected_strategy: str,
) -> None:
    path = _REPO_ROOT / "artifacts" / "experiment_log.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    required_fields = {
        "strategy",
        "chunking",
        "chunk_size",
        "overlap",
        "metadata",
        "filter",
        "vectorizer",
        "hypothesis",
        "dev_hit1",
        "dev_hit3",
        "test_hit1",
        "test_hit3",
        "notes",
    }
    _require(
        fieldnames is not None and required_fields.issubset(fieldnames),
        "Experiment log is missing required columns",
    )

    settings = {
        "A": {
            "chunking": "fixed_size",
            "chunk_size": "500",
            "overlap": "50",
            "filter": "none",
            "hypothesis": "Fixed overlapping windows may retain enough context for evidence retrieval.",
        },
        "B": {
            "chunking": "by_paragraph",
            "chunk_size": "500",
            "overlap": "",
            "filter": "none",
            "hypothesis": "Paragraph grouping may preserve coherent evidence within the same size limit.",
        },
        "C": {
            "chunking": "by_paragraph (same chunks as B)",
            "chunk_size": "500",
            "overlap": "",
            "filter": "question-provided category only",
            "hypothesis": "Category filtering may improve retrieval for questions that explicitly request it.",
        },
    }
    found: set[str] = set()
    for row in rows:
        strategy = row.get("strategy")
        if strategy not in settings:
            continue
        _require(strategy not in found, f"Duplicate strategy row in experiment log: {strategy}")
        found.add(strategy)
        config = settings[strategy]
        metrics = summary[strategy]["dev"]
        notes = [
            f"DEV only; {chunk_counts[strategy]} stored chunks.",
            f"Selected strategy: {selected_strategy}.",
        ]
        if strategy == "C":
            notes.append("C reuses B chunks/vectors.")
        row.update(
            {
                **config,
                "metadata": "document_id, source, page, category",
                "vectorizer": f"CKEY text-embedding-3-small ({dimensions} dimensions)",
                "dev_hit1": (
                    f"{metrics['hit@1']['hits']}/{metrics['hit@1']['denominator']} "
                    f"({metrics['hit@1']['rate']:.4f})"
                ),
                "dev_hit3": (
                    f"{metrics['hit@3']['hits']}/{metrics['hit@3']['denominator']} "
                    f"({metrics['hit@3']['rate']:.4f})"
                ),
                "test_hit1": "",
                "test_hit3": "",
                "notes": " ".join(notes),
            }
        )
    _require(found == set(_STRATEGIES), "Experiment log must have one row for A, B and C")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _redact_error(message: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(
        r"(?i)(postgres(?:ql)?://)[^@\s]+@",
        r"\1[REDACTED]@",
        message,
    )
    return " ".join(message.split())


def _read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"))
    _require(isinstance(value, dict), f"Expected a JSON object in {path.name}")
    return raw, value


def _frozen_test_inputs(
    env_values: dict[str, str | None],
    inputs: dict[str, Any],
    config: dict[str, Any],
    dev_metrics: dict[str, Any],
    dev_traces: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    model_name = env_values.get("EMBEDDING_MODEL")
    _require(
        isinstance(model_name, str) and model_name == config.get("embedding_model"),
        "Configured embedding model differs from frozen config.json",
    )
    _require(
        _safe_endpoint(env_values.get("OPENAI_BASE_URL"))
        == config.get("api_endpoint"),
        "Configured embedding endpoint differs from frozen config.json",
    )
    _require(config.get("provider") == "CKEY", "Frozen provider is not CKEY")
    dimensions = config.get("embedding_dimensions")
    _require(
        isinstance(dimensions, int) and not isinstance(dimensions, bool) and dimensions > 0,
        "Frozen embedding dimensions are invalid",
    )
    _require(
        config.get("vector_store")
        == {
            "engine": "PostgreSQL + pgvector",
            "table": "public.day6_chunks",
            "metric": "cosine",
            "search": "exact",
        },
        "Vector store configuration differs from the frozen lab setup",
    )
    _require(
        config.get("selection", {}).get("selected_strategy") == "B"
        and config.get("selection", {}).get("test_used_for_selection") is False,
        "Frozen DEV selection is missing or does not select B",
    )
    _require(
        dev_metrics.get("evaluated_split") == "dev"
        and dev_metrics.get("test_retrieval_performed") is False
        and dev_metrics.get("selection", {}).get("selected_strategy") == "B",
        "Saved DEV metrics do not match the frozen B selection",
    )
    _require(
        len(dev_traces) == 45
        and all(trace.get("split") == "dev" for trace in dev_traces),
        "Saved DEV traces are missing, malformed, or contain non-DEV traces",
    )
    _require(
        {
            strategy: sum(trace.get("strategy") == strategy for trace in dev_traces)
            for strategy in _STRATEGIES
        }
        == {strategy: 15 for strategy in _STRATEGIES}
        and summarize(dev_traces) == dev_metrics.get("retrieval_metrics"),
        "Saved DEV traces do not match the recorded DEV metrics",
    )
    _require(
        inputs["manifest_sha256"] == config["corpus"]["manifest_sha256"]
        and inputs["corpus_content_sha256"] == config["corpus"]["content_sha256"],
        "Corpus manifest or normalized content differs from frozen config.json",
    )
    _require(
        [doc.document_id for doc in inputs["documents"]]
        == config["corpus"]["source_ids"]
        and len(inputs["documents"]) == config["corpus"]["source_count"]
        and len(inputs["source_by_id"]) == config["corpus"]["unit_count"],
        "Corpus source IDs/count differ from frozen config.json",
    )
    _require(
        inputs["benchmark_sha256"] == config["benchmark"]["sha256"]
        and len(inputs["questions"]) == config["benchmark"]["question_count"]
        and inputs["split_counts"] == config["benchmark"]["split_counts"]
        and inputs["benchmark"]["selection_rule"]
        == config["benchmark"]["selection_rule"],
        "Benchmark differs from frozen config.json",
    )
    _require(
        [
            question["id"]
            for question in inputs["questions"]
            if question["split"] == "dev"
        ]
        == config["benchmark"]["dev_question_ids"],
        "DEV question IDs differ from frozen config.json",
    )
    test_questions = [
        question for question in inputs["questions"]
        if question.get("split") == "test"
    ]
    _require(len(test_questions) == 5, "TEST query set is not exactly five questions")
    _require(
        config["coherence_sample"] == _coherence_sample(inputs["documents"]),
        "Frozen coherence sample differs from the current corpus",
    )
    expected_strategies = {
        "A": {
            "chunking": "fixed_size",
            "parameters": {"size": 500, "overlap": 50},
            "filter": "none",
        },
        "B": {
            "chunking": "by_paragraph",
            "parameters": {"max_size": 500},
            "filter": "none",
        },
        "C": {
            "chunking": "by_paragraph",
            "parameters": {"max_size": 500},
            "filter": "question-provided category only",
            "reuse": "Same chunks, metadata and document vectors as B.",
        },
    }
    _require(
        config.get("strategies") == expected_strategies,
        "Chunk strategies or parameters differ from frozen config.json",
    )
    return {"questions": test_questions}, model_name


def _build_frozen_chunks(
    documents: list[DocumentPage],
) -> dict[str, list[Chunk]]:
    a_chunks: list[Chunk] = []
    b_chunks: list[Chunk] = []
    for document in documents:
        a_chunks.extend(fixed_size(document, size=500, overlap=50))
        b_chunks.extend(by_paragraph(document, max_size=500))
    return {"A": a_chunks, "B": b_chunks, "C": b_chunks}


def _validate_frozen_index(
    database_url: str,
    dimensions: int,
    documents: list[DocumentPage],
    chunks_by_strategy: dict[str, list[Chunk]],
    expected_counts: dict[str, int],
    expected_database_versions: dict[str, str],
) -> dict[str, Any]:
    expected_documents = {document.document_id for document in documents}
    expected_rows = {
        strategy: {
            chunk.chunk_id: (
                chunk.document_id,
                chunk.source,
                chunk.page,
                chunk.text,
                chunk.metadata,
            )
            for chunk in chunks_by_strategy[strategy]
        }
        for strategy in _STRATEGIES
    }
    _require(
        {strategy: len(rows) for strategy, rows in expected_rows.items()}
        == expected_counts,
        "Current chunking output does not match the frozen DEV chunk counts",
    )

    with psycopg.connect(database_url) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT to_regclass(%s)", ("public.day6_chunks",))
        _require(
            cursor.fetchone()[0] is not None,
            "Frozen vector table public.day6_chunks does not exist",
        )
        cursor.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute AS a
            WHERE a.attrelid = 'public.day6_chunks'::regclass
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
            """
        )
        dimension_type = cursor.fetchone()
        _require(
            dimension_type is not None
            and dimension_type[0] == f"vector({dimensions})",
            "Stored vector column dimension differs from frozen config.json",
        )
        cursor.execute(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = 'public' AND tablename = 'day6_chunks'
            """
        )
        index_definitions = [row[0] for row in cursor.fetchall()]
        _require(
            not any(
                re.search(r"\bUSING\s+(hnsw|ivfflat)\b", definition, re.IGNORECASE)
                for definition in index_definitions
            ),
            "Approximate vector index found; frozen config requires exact search",
        )
        cursor.execute(
            """
            SELECT strategy, chunk_id, document_id, source, page, text, metadata
            FROM public.day6_chunks
            WHERE strategy = ANY(%s)
            """,
            (list(_STRATEGIES),),
        )
        stored_rows = cursor.fetchall()
        cursor.execute(
            """
            SELECT count(*)
            FROM public.day6_chunks AS b
            FULL OUTER JOIN public.day6_chunks AS c
              ON b.strategy = 'B'
             AND c.strategy = 'C'
             AND b.chunk_id = c.chunk_id
            WHERE (b.strategy = 'B' OR c.strategy = 'C')
              AND (
                    b.chunk_id IS NULL OR c.chunk_id IS NULL
                 OR b.document_id IS DISTINCT FROM c.document_id
                 OR b.source IS DISTINCT FROM c.source
                 OR b.page IS DISTINCT FROM c.page
                 OR b.text IS DISTINCT FROM c.text
                 OR b.metadata IS DISTINCT FROM c.metadata
                 OR b.embedding IS DISTINCT FROM c.embedding
              )
            """
        )
        b_c_differences = cursor.fetchone()[0]
        cursor.execute(
            """
            SELECT count(*), count(*) FILTER (
                WHERE vector_dims(embedding) <> %s
            )
            FROM public.day6_chunks
            WHERE strategy = ANY(%s)
            """,
            (dimensions, list(_STRATEGIES)),
        )
        total_rows, invalid_dimension_rows = cursor.fetchone()
        cursor.execute(
            """
            SELECT current_setting('server_version'),
                   (SELECT extversion FROM pg_extension WHERE extname = 'vector')
            """
        )
        database_versions = {
            "postgresql": cursor.fetchone()[0],
        }
        cursor.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        pgvector_version = cursor.fetchone()
        database_versions["pgvector"] = (
            pgvector_version[0] if pgvector_version else None
        )

    _require(
        database_versions == expected_database_versions,
        "PostgreSQL/pgvector versions differ from saved DEV metrics",
    )
    actual_rows: dict[str, dict[str, tuple[Any, ...]]] = {
        strategy: {} for strategy in _STRATEGIES
    }
    actual_documents: dict[str, set[str]] = {
        strategy: set() for strategy in _STRATEGIES
    }
    for strategy, chunk_id, document_id, source, page, text, metadata in stored_rows:
        _require(
            chunk_id not in actual_rows[strategy],
            f"Duplicate stored chunk_id in strategy {strategy}: {chunk_id}",
        )
        actual_rows[strategy][chunk_id] = (
            document_id,
            source,
            page,
            text,
            metadata,
        )
        actual_documents[strategy].add(document_id)

    for strategy in _STRATEGIES:
        _require(
            len(actual_rows[strategy]) == expected_counts[strategy],
            f"Stored row count for {strategy} differs from frozen DEV metrics",
        )
        _require(
            actual_documents[strategy] == expected_documents,
            f"Stored document IDs for {strategy} differ from frozen corpus",
        )
        _require(
            actual_rows[strategy] == expected_rows[strategy],
            f"Stored chunks/metadata for {strategy} differ from frozen corpus/config",
        )
    _require(
        total_rows == sum(expected_counts.values())
        and invalid_dimension_rows == 0,
        "Stored strategy row count or vector dimensions are invalid",
    )
    _require(
        b_c_differences == 0,
        "B and C stored chunks, metadata, or vectors are not identical",
    )
    return {
        "table": "public.day6_chunks",
        "row_counts": {strategy: len(actual_rows[strategy]) for strategy in _STRATEGIES},
        "document_count_per_strategy": {
            strategy: len(actual_documents[strategy]) for strategy in _STRATEGIES
        },
        "vector_dimensions": dimensions,
        "b_c_identical_chunks_metadata_vectors": True,
        "approximate_vector_indexes": False,
        "database_versions": database_versions,
    }


def _source_spans(
    document: DocumentPage,
    chunks: list[Chunk],
) -> list[tuple[int, int, Chunk]]:
    spans = []
    search_from = 0
    for chunk in chunks:
        if chunk.document_id != document.document_id:
            continue
        start = document.text.find(chunk.text, search_from)
        if start < 0:
            raise ValueError(
                f"Cannot locate chunk {chunk.chunk_id} in source {document.document_id}"
            )
        end = start + len(chunk.text)
        spans.append((start, end, chunk))
        search_from = start + 1
    return spans


def _manual_coherence_rows(
    config: dict[str, Any],
    documents: list[DocumentPage],
    chunks_by_strategy: dict[str, list[Chunk]],
) -> list[dict[str, Any]]:
    documents_by_id = {document.document_id: document for document in documents}
    span_cache = {
        (strategy, document.document_id): _source_spans(
            document,
            [
                chunk for chunk in chunks_by_strategy[strategy]
                if chunk.document_id == document.document_id
            ],
        )
        for strategy in _STRATEGIES
        for document in documents
    }
    rows = []
    selection_rule = (
        "Maximum overlap with configured source interval; ties choose earliest "
        "chunk start, then chunk order."
    )
    for sample in config["coherence_sample"]["items"]:
        document = documents_by_id.get(sample["document_id"])
        _require(document is not None, f"Missing coherence source {sample['document_id']}")
        start = sample["character_start"]
        end = sample["character_end_exclusive"]
        _require(
            sample["page"] == document.page
            and sample["unit_id"] == document.metadata["unit_id"]
            and 0 <= start < end <= len(document.text),
            f"Invalid frozen coherence position for {document.document_id}",
        )
        for strategy in _STRATEGIES:
            overlaps = []
            for order, (chunk_start, chunk_end, chunk) in enumerate(
                span_cache[(strategy, document.document_id)]
            ):
                overlap = max(0, min(end, chunk_end) - max(start, chunk_start))
                if overlap:
                    overlaps.append(
                        (overlap, chunk_start, order, chunk_end, chunk)
                    )
            _require(
                bool(overlaps),
                f"No {strategy} chunk covers frozen position in {document.document_id}",
            )
            overlap, chunk_start, _, chunk_end, selected = min(
                overlaps,
                key=lambda item: (-item[0], item[1], item[2]),
            )
            context_start = max(0, chunk_start - 200)
            context_end = min(len(document.text), chunk_end + 200)
            rows.append(
                {
                    "strategy": strategy,
                    "sample_id": (
                        f"{sample['category']}_{sample['document_id']}_"
                        f"{start}_{end}"
                    ),
                    "category": sample["category"],
                    "document_id": document.document_id,
                    "unit_id": sample["unit_id"],
                    "page": sample["page"],
                    "source_position_start": start,
                    "source_position_end_exclusive": end,
                    "chunk_source_start": chunk_start,
                    "chunk_source_end_exclusive": chunk_end,
                    "overlap_characters": overlap,
                    "chunk_selection_rule": selection_rule,
                    "chunk_id": selected.chunk_id,
                    "chunk_text": selected.text,
                    "source_context_start": context_start,
                    "source_context_end_exclusive": context_end,
                    "source_context": document.text[context_start:context_end],
                    "score_0_to_2": "",
                    "reason": "",
                }
            )
    _require(len(rows) == 30, f"Expected 30 coherence rows, got {len(rows)}")
    return rows


def _manual_grounding_rows(
    traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for trace in traces:
        row = {
            "strategy": trace["strategy"],
            "question_id": trace["question_id"],
            "question": trace["question"],
            "expected_answer": trace["expected_answer"],
            "gold_source_id": trace["gold_source_id"],
            "gold_page": trace["gold_page"],
            "gold_unit_id": trace["gold_unit_id"],
            "evidence_quote": json.dumps(trace["evidence_quote"], ensure_ascii=False),
            "alternative_gold": json.dumps(
                trace.get("alternative_gold"), ensure_ascii=False
            ),
        }
        for rank in range(1, 4):
            hit = trace["top3"][rank - 1] if len(trace["top3"]) >= rank else {}
            for field in (
                "chunk_id",
                "document_id",
                "source",
                "page",
                "text",
                "score",
                "metadata",
            ):
                value = hit.get(field, "")
                row[f"top{rank}_{field}"] = (
                    json.dumps(value, ensure_ascii=False)
                    if field == "metadata" and value != ""
                    else value
                )
        row["score_0_to_2"] = ""
        row["reason"] = ""
        rows.append(row)
    _require(len(rows) == 15, f"Expected 15 grounding rows, got {len(rows)}")
    return rows


def _manual_metadata_rows(
    traces: list[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for trace in traces:
        requested_filter = trace.get("filter_requested") or {}
        applied_filter = trace.get("filter_applied") or {}
        for rank, hit in enumerate(trace["top3"], start=1):
            manifest_source = source_by_id.get(hit["document_id"])
            _require(
                manifest_source is not None,
                f"Top-3 document {hit['document_id']} is absent from manifest",
            )
            manifest_metadata = manifest_source["metadata"]
            manifest_unit = manifest_source["units"][0]
            rows.append(
                {
                    "strategy": trace["strategy"],
                    "question_id": trace["question_id"],
                    "rank": rank,
                    "chunk_id": hit["chunk_id"],
                    "document_id": hit["document_id"],
                    "source": hit["source"],
                    "page": hit["page"],
                    "unit_id": hit["metadata"].get("unit_id", ""),
                    "category": hit["metadata"].get("category", ""),
                    "requested_filter": json.dumps(
                        requested_filter, ensure_ascii=False, sort_keys=True
                    ),
                    "applied_filter": (
                        json.dumps(applied_filter, ensure_ascii=False, sort_keys=True)
                        if applied_filter
                        else "none"
                    ),
                    "filter_check_applicable": (
                        "yes" if requested_filter else "N/A (no requested filter)"
                    ),
                    "manifest_source_id": manifest_source["source_id"],
                    "manifest_title": manifest_source["title"],
                    "manifest_source_path_or_url": manifest_source[
                        "source_path_or_url"
                    ],
                    "manifest_publisher": manifest_metadata.get("source", ""),
                    "manifest_category": manifest_metadata.get("category", ""),
                    "manifest_unit_id": manifest_unit["unit_id"],
                    "manifest_page": manifest_unit["position"],
                    "manifest_local_path": manifest_source["local_path"],
                    "metadata_correct_complete": "",
                    "filter_compliance": "",
                    "reason": "",
                }
            )
    _require(len(rows) <= 45, f"Expected at most 45 metadata rows, got {len(rows)}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows), f"Cannot write empty audit table: {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _update_test_experiment_log(test_summary: dict[str, Any]) -> None:
    path = _REPO_ROOT / "artifacts" / "experiment_log.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    _require(
        fieldnames is not None
        and {"strategy", "dev_hit1", "dev_hit3", "test_hit1", "test_hit3"}.issubset(
            fieldnames
        ),
        "Experiment log is missing required columns",
    )
    found = set()
    for row in rows:
        strategy = row.get("strategy")
        if strategy not in _STRATEGIES:
            continue
        _require(strategy not in found, f"Duplicate strategy row in experiment log: {strategy}")
        found.add(strategy)
        metrics = test_summary[strategy]["test"]
        row["test_hit1"] = (
            f"{metrics['hit@1']['hits']}/{metrics['hit@1']['denominator']} "
            f"({metrics['hit@1']['rate']:.4f})"
        )
        row["test_hit3"] = (
            f"{metrics['hit@3']['hits']}/{metrics['hit@3']['denominator']} "
            f"({metrics['hit@3']['rate']:.4f})"
        )
    _require(found == set(_STRATEGIES), "Experiment log must have one row for A, B and C")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_log(path: Path, messages: list[str]) -> None:
    if not messages:
        return
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for message in messages:
            file.write(message.rstrip() + "\n")


def _guard_test_audits(results_dir: Path = _RESULTS_DIR) -> None:
    sensitive_names = {
        "reason",
        "review_status",
        "metadata_correct_complete",
        "filter_compliance",
    }
    reviewed = []
    for filename in (
        "manual_coherence.csv",
        "manual_grounding.csv",
        "manual_metadata.csv",
    ):
        path = results_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(
                    f"TEST refused before API, database, or file writes: "
                    f"{filename} has no readable CSV header."
                )
            sensitive_fields = [
                field
                for field in reader.fieldnames
                if (
                    field.strip().casefold().replace(" ", "_") in sensitive_names
                    or "score" in field.casefold()
                    or "label" in field.casefold()
                )
            ]
            for row in reader:
                populated = [
                    field
                    for field in sensitive_fields
                    if (row.get(field) or "").strip()
                ]
                if populated:
                    reviewed.append((filename, populated))
                    break
    if reviewed:
        details = "; ".join(
            f"{filename} ({', '.join(fields)})"
            for filename, fields in reviewed
        )
        raise ValueError(
            "TEST refused before API, database, or file writes because scored or "
            f"reviewed audit data already exists: {details}. Preserve these audit "
            "CSVs outside starter/results before intentionally regenerating them."
        )


def _run_test_pipeline() -> None:
    _guard_test_audits()
    run_started = time.perf_counter()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    errors_path = _RESULTS_DIR / "errors.log"
    env_values = dotenv_values(_STARTER_DIR / ".env")
    secrets = [
        value
        for key in ("OPENAI_API_KEY", "DATABASE_URL")
        if isinstance((value := env_values.get(key)), str)
    ]
    stage = "loading frozen TEST configuration"
    try:
        database_url = env_values.get("DATABASE_URL")
        _require(
            isinstance(database_url, str) and bool(database_url),
            "DATABASE_URL is missing from starter/.env",
        )
        config_bytes, config = _read_json(_RESULTS_DIR / "config.json")
        dev_metrics_bytes, dev_metrics = _read_json(
            _RESULTS_DIR / "metrics_dev.json"
        )
        dev_trace_path = _RESULTS_DIR / "traces_dev.jsonl"
        dev_traces = [
            json.loads(line)
            for line in dev_trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        input_paths = {
            "manifest": _STARTER_DIR / "data" / "manifest.json",
            "benchmark": _REPO_ROOT / "benchmark" / "questions.json",
        }
        frozen_hashes = {
            "config": hashlib.sha256(config_bytes).hexdigest(),
            "metrics_dev": hashlib.sha256(dev_metrics_bytes).hexdigest(),
            "traces_dev": hashlib.sha256(dev_trace_path.read_bytes()).hexdigest(),
            **{
                key: hashlib.sha256(path.read_bytes()).hexdigest()
                for key, path in input_paths.items()
            },
        }

        stage = "TEST corpus, benchmark, and frozen configuration validation"
        preflight_started = time.perf_counter()
        inputs = _preflight()
        preflight_seconds = time.perf_counter() - preflight_started
        frozen, model_name = _frozen_test_inputs(
            env_values, inputs, config, dev_metrics, dev_traces
        )
        test_questions = frozen["questions"]

        stage = "read-only frozen vector index validation"
        index_started = time.perf_counter()
        chunks_by_strategy = _build_frozen_chunks(inputs["documents"])
        expected_counts = {
            strategy: dev_metrics["chunk_statistics"][strategy]["chunk_count"]
            for strategy in _STRATEGIES
        }
        _require(
            expected_counts
            == dev_metrics["storage_check"]["after_identical_reload"],
            "Saved DEV chunk counts differ from the frozen index row counts",
        )
        index_state = _validate_frozen_index(
            database_url,
            config["embedding_dimensions"],
            inputs["documents"],
            chunks_by_strategy,
            expected_counts,
            dev_metrics["database_versions"],
        )
        index_validation_seconds = time.perf_counter() - index_started
        store = PgVectorStore(database_url, config["embedding_dimensions"])

        stage = "TEST query embedding"
        embedder = MyEmbedder(model_name)
        query_usage_start = _snapshot_usage(embedder)
        query_vectors: dict[str, list[float]] = {}
        query_embedding_seconds = {}
        for question in test_questions:
            started = time.perf_counter()
            vector = embedder.embed_query(question["question"])
            query_embedding_seconds[question["id"]] = time.perf_counter() - started
            _require(
                len(vector) == config["embedding_dimensions"],
                f"Live embedding dimension differs from frozen config for {question['id']}",
            )
            query_vectors[question["id"]] = vector
        _require(
            embedder.model_name == config["embedding_model"]
            and embedder.dimensions == config["embedding_dimensions"],
            "Live embedding model/dimensions differ from frozen config.json",
        )
        query_usage = _usage_report(embedder, query_usage_start)
        _require(
            len(query_vectors) == 5 and query_usage["sdk_request_calls"] == 5,
            "Expected exactly one embedding API request for each TEST query",
        )

        stage = "TEST retrieval"
        traces = []
        retrieval_seconds: Counter[str] = Counter()
        for question in test_questions:
            for strategy in _STRATEGIES:
                requested_filter = question.get("filter") or {}
                applied_filter = (
                    requested_filter
                    if strategy == "C" and requested_filter
                    else {}
                )
                started = time.perf_counter()
                hits = store.search(
                    strategy,
                    query_vectors[question["id"]],
                    top_k=3,
                    filters=applied_filter,
                )
                retrieval_seconds[strategy] += time.perf_counter() - started
                trace_question = dict(question)
                trace_question["strategy"] = strategy
                trace_question["filter_applied"] = applied_filter
                trace = evaluate_query(trace_question, hits)
                trace["config_sha256"] = frozen_hashes["config"]
                traces.append(trace)
        _require(
            len(traces) == 15
            and all(trace["split"] == "test" for trace in traces),
            f"Expected exactly 15 TEST traces, got {len(traces)}",
        )
        test_summary = summarize(traces)
        _require(
            set(test_summary) == set(_STRATEGIES)
            and all(test_summary[strategy]["test"]["N"] == 5 for strategy in _STRATEGIES),
            "TEST summary is incomplete",
        )

        stage = "manual audit table generation"
        coherence_rows = _manual_coherence_rows(
            config, inputs["documents"], chunks_by_strategy
        )
        grounding_rows = _manual_grounding_rows(traces)
        metadata_rows = _manual_metadata_rows(traces, inputs["source_by_id"])
        audit_guidance = {
            "manual_coherence.csv": (
                "Score 0=missing meaning/context seriously; 1=partly understandable; "
                "2=self-contained and preserves the complete idea."
            ),
            "manual_grounding.csv": (
                "Score 0=no evidence; 1=evidence is incomplete or lacks a condition/context; "
                "2=sufficient to answer and trace to source."
            ),
            "manual_metadata.csv": (
                "Leave metadata_correct_complete, filter_compliance and reason blank until "
                "human review. filter_check_applicable is N/A when no filter was requested."
            ),
        }
        for filename, rows in (
            ("manual_coherence.csv", coherence_rows),
            ("manual_grounding.csv", grounding_rows),
            ("manual_metadata.csv", metadata_rows),
        ):
            for row in rows:
                row["scoring_guidance"] = audit_guidance[filename]
        _write_csv(_RESULTS_DIR / "manual_coherence.csv", coherence_rows)
        _write_csv(_RESULTS_DIR / "manual_grounding.csv", grounding_rows)
        _write_csv(_RESULTS_DIR / "manual_metadata.csv", metadata_rows)

        stage = "TEST metrics and experiment log writing"
        _update_test_experiment_log(test_summary)
        test_run_elapsed = time.perf_counter() - run_started
        test_metrics = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "evaluated_split": "test",
            "test_retrieval_performed": True,
            "selection_changed": False,
            "selected_strategy_frozen": config["selection"]["selected_strategy"],
            "config_sha256": frozen_hashes["config"],
            "corpus_manifest_sha256": frozen_hashes["manifest"],
            "corpus_content_sha256": inputs["corpus_content_sha256"],
            "benchmark_sha256": frozen_hashes["benchmark"],
            "embedding_provider": config["provider"],
            "embedding_endpoint": config["api_endpoint"],
            "embedding_model": embedder.model_name,
            "embedding_dimensions": embedder.dimensions,
            "query_count": len(test_questions),
            "trace_count": len(traces),
            "retrieval_top_k": 3,
            "retrieval_metrics": test_summary,
            "resources": {
                "timings_seconds": {
                    "preflight_validation": preflight_seconds,
                    "read_only_index_validation": index_validation_seconds,
                    "query_embedding_shared_across_strategies": {
                        "total": sum(query_embedding_seconds.values()),
                        "per_question": query_embedding_seconds,
                    },
                    "retrieval_by_strategy": dict(retrieval_seconds),
                    "test_run_elapsed_including_validation_and_audit_export": (
                        test_run_elapsed
                    ),
                    "elapsed_time_scope": (
                        "TEST-only validation, query embedding, retrieval and audit export; "
                        "not corpus ingestion or the full DEV+TEST pipeline."
                    ),
                },
                "api_usage": {
                    "test_query_embeddings": query_usage,
                    "document_embeddings_performed": False,
                    "corpus_ingestion_performed": False,
                    "request_count_note": (
                        "Counts OpenAI SDK embeddings.create calls; SDK-internal HTTP retries "
                        "are not separately exposed."
                    ),
                    "cost_note": (
                        "Estimated from CKEY's listed input price/minimum charge, not provider "
                        "billing records."
                    ),
                },
            },
            "index_validation": index_state,
            "audit_row_counts": {
                "manual_coherence.csv": len(coherence_rows),
                "manual_grounding.csv": len(grounding_rows),
                "manual_metadata.csv": len(metadata_rows),
            },
        }
        with (_RESULTS_DIR / "traces_test.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as file:
            for trace in traces:
                file.write(json.dumps(trace, ensure_ascii=False) + "\n")
        _write_json(_RESULTS_DIR / "metrics_test.json", test_metrics)

        current_hashes = {
            "config": hashlib.sha256(
                (_RESULTS_DIR / "config.json").read_bytes()
            ).hexdigest(),
            "metrics_dev": hashlib.sha256(
                (_RESULTS_DIR / "metrics_dev.json").read_bytes()
            ).hexdigest(),
            "traces_dev": hashlib.sha256(
                (_RESULTS_DIR / "traces_dev.jsonl").read_bytes()
            ).hexdigest(),
            **{
                key: hashlib.sha256(path.read_bytes()).hexdigest()
                for key, path in input_paths.items()
            },
        }
        _require(
            current_hashes == frozen_hashes,
            "A frozen config, DEV artifact, corpus manifest, or benchmark changed during TEST",
        )

        log_messages = [
            "NOTE: TEST ran only five benchmark questions across A/B/C; no corpus document was embedded or ingested.",
            "NOTE: TEST elapsed time includes preflight, read-only index validation, query embedding, retrieval and audit generation/export; it is not ingest or full-pipeline time.",
        ]
        if query_usage["usage_available"]:
            log_messages.append(
                "NOTE: TEST API cost is an estimate based on CKEY's listed price and per-request minimum, not a billing record."
            )
        else:
            log_messages.append(
                "WARNING: TEST API usage/cost is unknown because "
                + query_usage["cost_unknown_reason"]
            )
        _append_log(errors_path, log_messages)

        print(
            "TEST complete: 5 queries; 15 traces; document embedding and ingestion not performed."
        )
        for strategy in _STRATEGIES:
            metrics_for_strategy = test_summary[strategy]["test"]
            print(
                f"{strategy}: Hit@1 {metrics_for_strategy['hit@1']['hits']}/5; "
                f"Evidence Hit@3 {metrics_for_strategy['hit@3']['hits']}/5"
            )
        print(
            f"Query SDK calls={query_usage['sdk_request_calls']}; prompt tokens="
            f"{query_usage['prompt_tokens']}; estimated API cost="
            f"{query_usage['estimated_cost_usd']} USD"
        )
        print(
            "Audit rows: "
            f"coherence={len(coherence_rows)}, grounding={len(grounding_rows)}, "
            f"metadata={len(metadata_rows)}; frozen selection remains B."
        )
    except Exception as exc:
        safe_message = _redact_error(str(exc) or type(exc).__name__, secrets)
        _append_log(
            errors_path,
            [f"ERROR during TEST {stage}: {type(exc).__name__}: {safe_message}"],
        )
        raise RuntimeError(
            f"TEST stopped during {stage}: {safe_message}"
        ) from None


def _run_pipeline() -> None:
    run_started = time.perf_counter()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    errors_path = _RESULTS_DIR / "errors.log"
    errors: list[str] = []
    stage = "loading configuration"
    env_values = dotenv_values(_STARTER_DIR / ".env")
    secrets = [
        value
        for key in ("OPENAI_API_KEY", "DATABASE_URL")
        if isinstance((value := env_values.get(key)), str)
    ]

    try:
        _require(bool(env_values.get("DATABASE_URL")), "DATABASE_URL is missing from starter/.env")
        model_name = env_values.get("EMBEDDING_MODEL")
        _require(isinstance(model_name, str) and bool(model_name), "EMBEDDING_MODEL is missing from starter/.env")

        stage = "manifest and benchmark validation"
        preflight_start = time.perf_counter()
        inputs = _preflight()
        preflight_seconds = time.perf_counter() - preflight_start
        documents: list[DocumentPage] = inputs["documents"]
        questions = inputs["questions"]
        dev_questions = [question for question in questions if question["split"] == "dev"]
        _require(len(dev_questions) == 15, "DEV query set is not exactly 15 questions")

        stage = "chunk generation"
        chunking_times = {}
        a_chunks: list[Chunk] = []
        b_chunks: list[Chunk] = []
        started = time.perf_counter()
        for document in documents:
            a_chunks.extend(fixed_size(document, size=500, overlap=50))
        chunking_times["A"] = time.perf_counter() - started
        started = time.perf_counter()
        for document in documents:
            b_chunks.extend(by_paragraph(document, max_size=500))
        chunking_times["B"] = time.perf_counter() - started
        c_chunks = b_chunks
        chunking_times["C"] = 0.0
        chunks_by_strategy = {"A": a_chunks, "B": b_chunks, "C": c_chunks}
        chunk_stats = {
            strategy: _chunk_statistics(chunks_by_strategy[strategy], documents)
            for strategy in _STRATEGIES
        }
        chunk_counts = {
            strategy: chunk_stats[strategy]["chunk_count"]
            for strategy in _STRATEGIES
        }
        _require(
            all(chunk_counts[strategy] > 0 for strategy in _STRATEGIES),
            "Chunking produced an empty strategy",
        )

        stage = "embedding client initialization and dimensions"
        embedder = MyEmbedder(model_name)
        dimension_usage_start = _snapshot_usage(embedder)
        started = time.perf_counter()
        dimensions = embedder.dimensions
        dimension_probe_seconds = time.perf_counter() - started
        dimension_usage = _usage_report(embedder, dimension_usage_start)
        database_url = env_values["DATABASE_URL"]
        store = PgVectorStore(database_url, dimensions)
        coherence_sample = _coherence_sample(documents)
        initial_config = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phase": "DEV only; TEST retrieval and metrics are not run",
            "provider": "CKEY",
            "api_endpoint": _safe_endpoint(env_values.get("OPENAI_BASE_URL")),
            "embedding_model": embedder.model_name,
            "embedding_dimensions": dimensions,
            "vector_store": {
                "engine": "PostgreSQL + pgvector",
                "table": "public.day6_chunks",
                "metric": "cosine",
                "search": "exact",
            },
            "versions": _package_versions(),
            "strategies": {
                "A": {
                    "chunking": "fixed_size",
                    "parameters": {"size": 500, "overlap": 50},
                    "filter": "none",
                },
                "B": {
                    "chunking": "by_paragraph",
                    "parameters": {"max_size": 500},
                    "filter": "none",
                },
                "C": {
                    "chunking": "by_paragraph",
                    "parameters": {"max_size": 500},
                    "filter": "question-provided category only",
                    "reuse": "Same chunks, metadata and document vectors as B.",
                },
            },
            "corpus": {
                "manifest_path": "starter/data/manifest.json",
                "manifest_sha256": inputs["manifest_sha256"],
                "content_sha256": inputs["corpus_content_sha256"],
                "source_count": len(documents),
                "unit_count": len(inputs["source_by_id"]),
                "category_counts": inputs["category_counts"],
                "source_ids": [document.document_id for document in documents],
            },
            "benchmark": {
                "path": "benchmark/questions.json",
                "sha256": inputs["benchmark_sha256"],
                "question_count": len(questions),
                "split_counts": inputs["split_counts"],
                "dev_question_ids": [question["id"] for question in dev_questions],
                "selection_rule": inputs["benchmark"]["selection_rule"],
            },
            "coherence_sample": coherence_sample,
            "selection": {
                "status": "pending DEV results",
                "criteria": [
                    "Evidence Hit@3 descending",
                    "Hit@1 descending",
                    "stored chunk count ascending",
                    "fixed order A, B, C",
                ],
                "test_used_for_selection": False,
            },
            "pricing": {
                "source_url": _PRICE_URL,
                "input_usd_per_million_tokens": _INPUT_PRICE_PER_MILLION,
                "minimum_usd_per_request": _MIN_PRICE_PER_REQUEST,
                "estimate_method": "Per SDK response: max(prompt-token price, minimum request charge).",
            },
        }
        _write_json(_RESULTS_DIR / "config.json", initial_config)

        stage = "PostgreSQL schema initialization"
        store_started = time.perf_counter()
        store.create_schema()
        schema_seconds = time.perf_counter() - store_started
        database_versions = _database_info(database_url)

        stage = "A document embedding"
        a_usage_start = _snapshot_usage(embedder)
        started = time.perf_counter()
        a_vectors = embedder.embed_documents([chunk.text for chunk in a_chunks])
        a_embedding_seconds = time.perf_counter() - started
        a_usage = _usage_report(embedder, a_usage_start)
        _require(len(a_vectors) == len(a_chunks), "A embedding count does not match A chunks")
        _require(
            all(len(vector) == dimensions for vector in a_vectors),
            "A embedding dimension does not match the live model dimension",
        )

        stage = "B document embedding"
        b_usage_start = _snapshot_usage(embedder)
        started = time.perf_counter()
        b_vectors = embedder.embed_documents([chunk.text for chunk in b_chunks])
        b_embedding_seconds = time.perf_counter() - started
        b_usage = _usage_report(embedder, b_usage_start)
        _require(len(b_vectors) == len(b_chunks), "B embedding count does not match B chunks")
        _require(
            all(len(vector) == dimensions for vector in b_vectors),
            "B embedding dimension does not match the live model dimension",
        )

        stage = "DEV query embedding"
        query_usage_start = _snapshot_usage(embedder)
        query_vectors: dict[str, list[float]] = {}
        query_embedding_seconds = {}
        for question in dev_questions:
            started = time.perf_counter()
            query_vector = embedder.embed_query(question["question"])
            query_embedding_seconds[question["id"]] = time.perf_counter() - started
            _require(
                len(query_vector) == dimensions,
                f"Query embedding dimension mismatch for {question['id']}",
            )
            query_vectors[question["id"]] = query_vector
        query_usage = _usage_report(embedder, query_usage_start)
        _require(
            len(query_vectors) == len(dev_questions),
            "Not all DEV questions received exactly one query vector",
        )

        stage = "strategy index refresh"
        document_ids = [document.document_id for document in documents]
        pruned_rows = _remove_obsolete_strategy_rows(database_url, document_ids)
        vectors_by_strategy = {"A": a_vectors, "B": b_vectors, "C": b_vectors}
        storage_seconds = {}
        for strategy in _STRATEGIES:
            started = time.perf_counter()
            store.upsert(
                strategy,
                chunks_by_strategy[strategy],
                vectors_by_strategy[strategy],
            )
            storage_seconds[strategy] = time.perf_counter() - started
        initial_row_counts = _strategy_row_counts(database_url)
        expected_counts = dict(chunk_counts)
        _require(
            initial_row_counts == expected_counts,
            f"Stored row counts do not match chunk counts: {initial_row_counts}",
        )

        stage = "idempotent re-ingestion check"
        reload_seconds = {}
        for strategy in _STRATEGIES:
            started = time.perf_counter()
            store.upsert(
                strategy,
                chunks_by_strategy[strategy],
                vectors_by_strategy[strategy],
            )
            reload_seconds[strategy] = time.perf_counter() - started
        reloaded_row_counts = _strategy_row_counts(database_url)
        _require(
            reloaded_row_counts == initial_row_counts,
            "Re-ingesting identical chunks/vectors changed stored row counts",
        )
        foreign_documents = _foreign_strategy_documents(database_url, document_ids)
        _require(
            not foreign_documents,
            f"Unexpected non-corpus documents remain in A/B/C: {foreign_documents}",
        )

        stage = "DEV retrieval"
        traces = []
        retrieval_seconds = Counter()
        for question in dev_questions:
            for strategy in _STRATEGIES:
                requested_filter = question.get("filter") or {}
                applied_filter = (
                    requested_filter
                    if strategy == "C" and requested_filter
                    else {}
                )
                started = time.perf_counter()
                hits = store.search(
                    strategy,
                    query_vectors[question["id"]],
                    top_k=3,
                    filters=applied_filter,
                )
                retrieval_seconds[strategy] += time.perf_counter() - started
                trace_question = dict(question)
                trace_question["strategy"] = strategy
                trace_question["filter_applied"] = applied_filter
                traces.append(evaluate_query(trace_question, hits))

        _require(len(traces) == 45, f"Expected 45 DEV traces, got {len(traces)}")
        _require(
            all(trace["split"] == "dev" for trace in traces),
            "A non-DEV trace was produced",
        )
        dev_summary = summarize(traces)
        _require(
            set(dev_summary) == set(_STRATEGIES)
            and all(dev_summary[strategy]["dev"]["N"] == 15 for strategy in _STRATEGIES),
            "DEV summary is incomplete",
        )

        stage = "DEV strategy selection"
        selected_strategy, selection = _select_strategy(dev_summary, chunk_counts)
        config = dict(initial_config)
        config["selection"] = selection

        stage = "metrics and artifact writing"
        total_usage = _usage_report(embedder, (0, 0))
        total_embedding_seconds = (
            dimension_probe_seconds
            + a_embedding_seconds
            + b_embedding_seconds
            + sum(query_embedding_seconds.values())
        )
        resource_usage = {
            "timings_seconds": {
                "preflight_validation": preflight_seconds,
                "chunk_generation": chunking_times,
                "chunk_generation_note": (
                    "C reused B chunks; no independent chunking run occurred."
                ),
                "dimension_probe": dimension_probe_seconds,
                "document_embedding": {
                    "A": a_embedding_seconds,
                    "B": b_embedding_seconds,
                    "C": 0.0,
                    "C_note": "C reused B document vectors; no independent C embedding run occurred.",
                },
                "query_embedding_shared_across_strategies": {
                    "total": sum(query_embedding_seconds.values()),
                    "per_question": query_embedding_seconds,
                },
                "schema_initialization": schema_seconds,
                "storage_initial_upsert": storage_seconds,
                "storage_idempotent_reload": reload_seconds,
                "retrieval_by_strategy": dict(retrieval_seconds),
                "embedding_total": total_embedding_seconds,
                "elapsed_through_retrieval_and_selection": (
                    time.perf_counter() - run_started
                ),
            },
            "api_usage": {
                "dimension_probe": dimension_usage,
                "document_embeddings": {
                    "A": a_usage,
                    "B": b_usage,
                    "C": {
                        "sdk_request_calls": 0,
                        "usage_available": True,
                        "prompt_tokens": 0,
                        "total_tokens": 0,
                        "estimated_cost_usd": 0.0,
                        "note": "No separate C document embedding; exact B vectors were reused.",
                    },
                },
                "dev_query_embeddings_shared": query_usage,
                "total": total_usage,
                "request_count_note": (
                    "Counts calls to OpenAI SDK embeddings.create; SDK-internal HTTP retries "
                    "are not separately exposed."
                ),
                "cost_note": (
                    "Estimated from CKEY's listed input price/minimum charge, not provider billing records."
                ),
            },
        }
        metrics = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "evaluated_split": "dev",
            "test_retrieval_performed": False,
            "query_count": 15,
            "trace_count": len(traces),
            "retrieval_metrics": dev_summary,
            "chunk_statistics": chunk_stats,
            "resources": resource_usage,
            "database_versions": database_versions,
            "storage_check": {
                "table": "public.day6_chunks",
                "initial_row_counts": initial_row_counts,
                "after_identical_reload": reloaded_row_counts,
                "identical_reload_did_not_increase_rows": True,
                "obsolete_non_corpus_rows_pruned_from_A_B_C": pruned_rows,
            },
            "selection": selection,
        }
        _write_json(_RESULTS_DIR / "config.json", config)
        with (_RESULTS_DIR / "traces_dev.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as file:
            for trace in traces:
                file.write(json.dumps(trace, ensure_ascii=False) + "\n")
        _write_json(_RESULTS_DIR / "metrics_dev.json", metrics)
        _update_experiment_log(dev_summary, chunk_counts, dimensions, selected_strategy)

        if not total_usage["usage_available"]:
            errors.append(
                "WARNING: One or more CKEY embedding responses omitted usage; token/cost totals are unknown."
            )
        errors.append(
            "NOTE: API cost is an estimate based on CKEY's listed price and per-request minimum, not a billing record."
        )
        errors.append(
            "NOTE: C reused B's document vectors; C embedding time and cost are incremental (not independent-run) values."
        )
        errors_path.write_text("\n".join(errors) + "\n", encoding="utf-8")

        print(
            f"DEV complete: {len(documents)} articles; chunks "
            + ", ".join(f"{strategy}={chunk_counts[strategy]}" for strategy in _STRATEGIES)
        )
        for strategy in _STRATEGIES:
            metrics_for_strategy = dev_summary[strategy]["dev"]
            print(
                f"{strategy}: Hit@1 {metrics_for_strategy['hit@1']['hits']}/15; "
                f"Evidence Hit@3 {metrics_for_strategy['hit@3']['hits']}/15"
            )
        print(
            f"Selected {selected_strategy} by DEV rule; re-ingestion row counts unchanged: "
            f"{reloaded_row_counts == initial_row_counts}"
        )
        print(
            f"Embedding SDK calls={total_usage['sdk_request_calls']}; prompt tokens="
            f"{total_usage['prompt_tokens']}; estimated API cost="
            f"{total_usage['estimated_cost_usd']} USD"
        )
        print("TEST retrieval and metrics were not run.")
    except Exception as exc:
        safe_message = _redact_error(str(exc) or type(exc).__name__, secrets)
        errors.append(f"ERROR during {stage}: {type(exc).__name__}: {safe_message}")
        errors_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"Pipeline stopped during {stage}: {safe_message}"
        ) from None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 06 retrieval experiment.")
    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="dev",
        help="Benchmark split to run (default: dev).",
    )
    args = parser.parse_args()
    if args.split == "test":
        _run_test_pipeline()
    else:
        _run_pipeline()


if __name__ == "__main__":
    main()
