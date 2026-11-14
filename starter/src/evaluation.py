"""Evaluate retrieved evidence and summarize retrieval hit rates."""
import unicodedata
from typing import Any


_HIT_FIELDS = (
    "chunk_id",
    "document_id",
    "source",
    "page",
    "text",
    "score",
    "metadata",
)


def _normalize_evidence(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.split())


def _quotes(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item for item in value if item]
    return []


def _gold_candidates(question: dict[str, Any]) -> list[dict[str, Any]]:
    primary = {
        "label": "primary",
        "source_id": question.get("gold_source_id"),
        "page": question.get("gold_page"),
        "unit_id": question.get("gold_unit_id"),
        "evidence_quotes": _quotes(question.get("evidence_quote")),
    }
    candidates = [primary]
    alternatives = question.get("alternative_gold", [])
    if isinstance(alternatives, (str, dict)):
        alternatives = [alternatives]
    if not isinstance(alternatives, list):
        raise ValueError("alternative_gold must be a string, object, or list")

    for index, alternative in enumerate(alternatives):
        label = f"alternative_gold[{index}]"
        if isinstance(alternative, str):
            candidate = dict(primary)
            candidate["label"] = label
            candidate["evidence_quotes"] = _quotes(alternative)
        elif isinstance(alternative, dict):
            candidate = {
                "label": label,
                "source_id": alternative.get(
                    "gold_source_id",
                    alternative.get("source_id", alternative.get("document_id", primary["source_id"])),
                ),
                "page": alternative.get(
                    "gold_page", alternative.get("page", primary["page"])
                ),
                "unit_id": alternative.get(
                    "gold_unit_id", alternative.get("unit_id", primary["unit_id"])
                ),
                "evidence_quotes": _quotes(
                    alternative.get(
                        "evidence_quote",
                        alternative.get(
                            "evidence_quotes",
                            alternative.get("evidence", primary["evidence_quotes"]),
                        ),
                    )
                ),
            }
        else:
            raise ValueError(
                "alternative_gold entries must be strings or objects"
            )
        candidates.append(candidate)
    return candidates


def _evaluate_hit(
    hit: dict[str, Any], gold_candidates: list[dict[str, Any]]
) -> tuple[bool, str, str | None]:
    document_candidates = [
        candidate
        for candidate in gold_candidates
        if candidate["source_id"] is not None
        and hit.get("document_id") == candidate["source_id"]
    ]
    if not document_candidates:
        return False, "document_id did not match any gold_source_id", None

    source_page_candidates = [
        candidate
        for candidate in document_candidates
        if candidate["page"] is not None and hit.get("page") == candidate["page"]
    ]
    if not source_page_candidates:
        return False, "document_id matched gold_source_id, but page did not match", None

    text = hit.get("text")
    normalized_text = _normalize_evidence(text) if isinstance(text, str) else ""
    for candidate in source_page_candidates:
        for quote in candidate["evidence_quotes"]:
            normalized_quote = _normalize_evidence(quote)
            if normalized_quote and normalized_quote in normalized_text:
                return (
                    True,
                    f"Declared evidence quote matched after NFC/whitespace normalization ({candidate['label']})",
                    candidate["label"],
                )

    if any(candidate["evidence_quotes"] for candidate in source_page_candidates):
        reason = (
            "document_id and page matched gold, but no declared evidence quote "
            "matched; equivalent wording requires human review"
        )
    else:
        reason = "No non-empty evidence quote is declared for the matching gold"
    return False, reason, None


def evaluate_query(
    question: dict[str, Any], hits: list[dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate exact evidence matches in the first three retrieved chunks."""
    gold_candidates = _gold_candidates(question)
    top3 = []
    for hit in hits[:3]:
        evidence_hit, reason, matched_gold = _evaluate_hit(hit, gold_candidates)
        traced_hit = {field: hit.get(field) for field in _HIT_FIELDS}
        traced_hit.update(
            {
                "evidence_hit": evidence_hit,
                "evidence_match_reason": reason,
                "matched_gold": matched_gold,
            }
        )
        top3.append(traced_hit)

    question_id = question.get("id")
    if question_id is None:
        question_id = question.get("question_id")
    strategy = question.get("strategy")
    filter_applied = question.get("filter_applied")
    if "filter_applied" not in question:
        filter_applied = question.get("applied_filter")
    gold = {
        "gold_source_id": question.get("gold_source_id"),
        "gold_page": question.get("gold_page"),
        "gold_unit_id": question.get("gold_unit_id"),
    }
    return {
        "id": question_id,
        "question_id": question_id,
        "split": question.get("split"),
        "strategy": strategy,
        "question": question.get("question"),
        "expected_answer": question.get("expected_answer"),
        "gold": gold,
        **gold,
        "evidence_quote": question.get("evidence_quote"),
        "alternative_gold": question.get("alternative_gold"),
        "filter_requested": question.get("filter"),
        "filter_applied": filter_applied,
        "top3": top3,
        "hit@1": bool(top3 and top3[0]["evidence_hit"]),
        "hit@3": any(hit["evidence_hit"] for hit in top3),
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate unique query hits by explicitly supplied strategy and split."""
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    seen: set[tuple[str, str, Any]] = set()

    for trace in traces:
        strategy = trace.get("strategy")
        split = trace.get("split")
        question_id = trace.get("question_id")
        if question_id is None:
            question_id = trace.get("id")
        if not isinstance(strategy, str) or not strategy:
            raise ValueError("Each trace must include a strategy")
        if split not in {"dev", "test"}:
            raise ValueError("Each trace split must be 'dev' or 'test'")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Each trace must include a non-empty question ID")

        group_key = (strategy, split, question_id)
        if group_key in seen:
            raise ValueError(
                f"Duplicate question ID {question_id!r} for "
                f"strategy {strategy!r}, split {split!r}"
            )
        seen.add(group_key)

        hit1 = trace.get("hit@1")
        hit3 = trace.get("hit@3")
        if not isinstance(hit1, bool) or not isinstance(hit3, bool):
            raise ValueError("Each trace must have boolean hit@1 and hit@3 values")

        group = groups.setdefault(strategy, {}).setdefault(
            split, {"N": 0, "hit@1": 0, "hit@3": 0}
        )
        group["N"] += 1
        group["hit@1"] += int(hit1)
        group["hit@3"] += int(hit3)

    summary = {}
    for strategy, splits in groups.items():
        summary[strategy] = {}
        for split, counts in splits.items():
            total = counts["N"]
            hit1 = counts["hit@1"]
            hit3 = counts["hit@3"]
            summary[strategy][split] = {
                "N": total,
                "hit@1": {
                    "hits": hit1,
                    "denominator": total,
                    "rate": hit1 / total,
                },
                "hit@3": {
                    "hits": hit3,
                    "denominator": total,
                    "rate": hit3 / total,
                },
            }
    return summary
