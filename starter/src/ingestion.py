"""Read documents, normalize text and preserve source positions."""
import logging
from pathlib import Path
import re
import unicodedata

from pypdf import PdfReader

from .models import DocumentPage


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        re.sub(r"[^\S\n]+", " ", line).strip()
        for line in text.split("\n")
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    pages = []
    for number, pdf_page in enumerate(PdfReader(path).pages, start=1):
        try:
            text = normalize_text(pdf_page.extract_text() or "")
        except Exception:
            logging.exception("Cannot extract %s, page %s", path, number)
            raise

        if not text:
            logging.warning("No extracted text: %s, page %s", path, number)

        pages.append(DocumentPage(
            document_id=document_id,
            source=source,
            page=number,
            text=text,
            metadata={"local_path": str(path), "empty": not bool(text)},
        ))
    return pages


def load_markdown(path: Path) -> DocumentPage:
    """Read one independent article from the selected Markdown corpus."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    separator = next(
        (i for i, line in enumerate(lines) if line.strip() == "---"),
        None,
    )
    if separator is None:
        raise ValueError(f"Missing metadata/body separator: {path}")

    header = "\n".join(lines[:separator])
    title = re.search(r"^#\s+(.+)$", header, re.MULTILINE)
    metadata = {
        key.lower(): value.strip()
        for key, value in re.findall(
            r"^\*\*(\w+)\*\*:[ \t]*(.*?)[ \t]*$",
            header,
            re.MULTILINE,
        )
    }

    required = ("category", "source", "url")
    if title is None or any(not metadata.get(key) for key in required):
        raise ValueError(f"Missing title/category/source/url: {path}")

    if metadata["category"] != path.parent.name:
        raise ValueError(f"Category disagrees with folder: {path}")

    start, end = separator + 1, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1

    text = normalize_text("\n".join(lines[start:end]))
    if not text:
        raise ValueError(f"Empty article body: {path}")

    match = re.match(r"(doc_\d+)_", path.name)
    if match is None:
        raise ValueError(f"Missing document ID: {path}")

    document_id = match.group(1)
    metadata.update({
        "title": title.group(1).strip(),
        "local_path": path.as_posix(),
        "unit_id": f"{document_id}:unit_1",
        "position_type": "independent_article",
        "start_line": start + 1,
        "end_line": end,
        "char_count": len(text),
    })

    # For Markdown, page=1 identifies the article's single source unit.
    # It does not mean a physical PDF page.
    return DocumentPage(
        document_id=document_id,
        source=metadata["url"],
        page=1,
        text=text,
        metadata=metadata,
    )
