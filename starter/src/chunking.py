"""Split source pages into traceable, deterministic chunks."""
import hashlib
import json
import re

from .models import Chunk, DocumentPage


def _chunk_id(
    page: DocumentPage,
    method: str,
    parameters: dict[str, int],
    position: int,
    start: int,
    end: int,
    text: str,
) -> str:
    identity = json.dumps(
        {
            "document_id": page.document_id,
            "source": page.source,
            "page": page.page,
            "method": method,
            "parameters": parameters,
            "position": position,
            "start": start,
            "end": end,
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"{method}-{digest}"


def _make_chunk(
    page: DocumentPage,
    method: str,
    parameters: dict[str, int],
    position: int,
    start: int,
    end: int,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id=_chunk_id(
            page, method, parameters, position, start, end, text
        ),
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=page.metadata.copy(),
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split text into fixed windows, retaining source metadata."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    text = page.text
    chunks = []
    start = 0
    parameters = {"size": size, "overlap": overlap}
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(
            _make_chunk(
                page, "fixed_size", parameters, len(chunks), start, end,
                text[start:end],
            )
        )
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _paragraph_spans(text: str) -> list[tuple[str, int, int]]:
    spans = []
    cursor = 0
    separator = re.compile(r"(?:\r?\n)[ \t]*(?:\r?\n)+")
    for match in separator.finditer(text):
        start, end = cursor, match.start()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((text[start:end], start, end))
        cursor = match.end()

    start, end = cursor, len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append((text[start:end], start, end))
    return spans


def _split_oversized(
    text: str, start: int, max_size: int
) -> list[tuple[str, int, int]]:
    pieces = []
    offset = 0
    while offset < len(text):
        end = min(offset + max_size, len(text))
        if end < len(text):
            boundary = next(
                (
                    index + 1
                    for index in range(end - 1, offset, -1)
                    if text[index].isspace()
                ),
                end,
            )
            end = boundary
        pieces.append((text[offset:end], start + offset, start + end))
        offset = end
    return pieces


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Pack adjacent paragraphs within the limit and split oversized ones."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than zero")

    paragraphs = _paragraph_spans(page.text)
    packed: list[tuple[str, int, int]] = []
    current_text = ""
    current_start = 0
    current_end = 0

    def flush() -> None:
        nonlocal current_text
        if current_text:
            packed.append((current_text, current_start, current_end))
            current_text = ""

    for paragraph, start, end in paragraphs:
        if len(paragraph) > max_size:
            flush()
            packed.extend(_split_oversized(paragraph, start, max_size))
            continue

        candidate = (
            paragraph
            if not current_text
            else f"{current_text}\n\n{paragraph}"
        )
        if current_text and len(candidate) > max_size:
            flush()
            candidate = paragraph

        if not current_text:
            current_start = start
        current_text = candidate
        current_end = end

    flush()
    parameters = {"max_size": max_size}
    return [
        _make_chunk(
            page, "by_paragraph", parameters, position, start, end, text
        )
        for position, (text, start, end) in enumerate(packed)
    ]
