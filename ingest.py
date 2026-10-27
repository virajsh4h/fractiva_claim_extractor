"""PDF ingestion with page provenance.

The module never substitutes a hand-written transcript for a real page. A page
is either native text, a mixed text/image page, or a scanned page that is sent
to the multimodal model. Every page carries its source label and PDF page
number so later evidence checks can point back to the exact document.
"""

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF

from domain import SourceRef

_GENERIC_REF_TOKENS = {
    "pdf",
    "paper",
    "papers",
    "the",
    "of",
    "and",
    "plus",
    "trial",
    "study",
    "abstract",
    "article",
}


def _tokens(value: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in _GENERIC_REF_TOKENS
    ]


def _pdf_candidates(base_dir: str) -> List[str]:
    directories = [
        os.path.join(base_dir, "papers"),
        os.path.join(os.getcwd(), "papers"),
        base_dir,
        os.getcwd(),
    ]
    seen = set()
    candidates: List[str] = []
    for directory in directories:
        directory = os.path.abspath(directory)
        if directory in seen or not os.path.isdir(directory):
            continue
        seen.add(directory)
        try:
            for name in os.listdir(directory):
                path = os.path.join(directory, name)
                if os.path.isfile(path) and name.lower().endswith(".pdf"):
                    candidates.append(path)
        except OSError:
            continue
    return candidates


def _alias_match(clean_ref: str, base_dir: str) -> Optional[str]:
    wanted = set(_tokens(clean_ref))
    if not wanted:
        return None

    scored: List[Tuple[float, int, str]] = []
    for path in _pdf_candidates(base_dir):
        have = set(_tokens(os.path.basename(path)))
        overlap = len(wanted & have)
        if not overlap:
            continue
        score = overlap / float(len(wanted))
        if score >= 0.6:
            scored.append((score, -len(have), path))

    if not scored:
        return None
    scored.sort(reverse=True)
    best = scored[0]
    if len(scored) > 1 and scored[1][0] == best[0]:
        # Ambiguous aliases are safer unresolved than silently mapped to the
        # wrong paper.
        return None
    return os.path.abspath(best[2])


def resolve_pdf_path(paper_ref: str, base_dir: str = ".") -> str:
    """Resolve one path or a human-readable paper alias to a local PDF."""

    clean_ref = re.sub(r"\(.*?\)", "", paper_ref or "").strip()
    if not clean_ref:
        return ""

    candidates = [
        os.path.join(base_dir, clean_ref),
        os.path.join(base_dir, "papers", os.path.basename(clean_ref)),
        os.path.join(os.getcwd(), clean_ref),
        os.path.join(os.getcwd(), "papers", os.path.basename(clean_ref)),
        clean_ref,
        os.path.basename(clean_ref),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)

    alias = _alias_match(clean_ref, base_dir)
    return alias if alias else clean_ref


def requested_sources(cited_paper: str) -> List[str]:
    """Split a cited-paper string into independent source requests.

    Handles ``A.pdf (plus B.pdf and C.pdf)`` and plain aliases. Parenthetical
    page annotations such as ``(Prescribing Information, page 4)`` or
    ``(see page 4)`` are not sources. A parenthetical is kept as a source only
    when it names a PDF or resolves to a local paper.
    """

    text = (cited_paper or "").strip()
    labels: List[str] = []

    # Every explicit PDF name is a source, including one inside parentheses.
    for match in re.finditer(r"[A-Za-z0-9][^,;()]*?\.pdf", text, flags=re.IGNORECASE):
        labels.append(match.group(0).strip(" ."))

    main = re.sub(r"\(.*?\)", "", text).strip()
    main = re.sub(r"^(?:see|from)\s+", "", main, flags=re.IGNORECASE)
    if main and not main.lower().endswith(".pdf"):
        labels.insert(0, main)

    annotation = re.compile(
        r"^\s*(?:see\s+)?(?:page|pp?\.|table|fig(?:ure)?|section|appendix|"
        r"prescribing|abridged|published|accessed)\b",
        flags=re.IGNORECASE,
    )
    for group in re.findall(r"\(([^)]*)\)", text):
        if annotation.match(group):
            continue
        cleaned = re.sub(r"^\s*(?:plus|and)\s+", "", group, flags=re.IGNORECASE)
        for part in re.split(r"\bplus\b|;|,|\band\b", cleaned, flags=re.IGNORECASE):
            part = part.strip()
            if not part or annotation.match(part):
                continue
            if part.lower().endswith(".pdf") or os.path.isfile(resolve_pdf_path(part)):
                labels.append(part)

    if not labels:
        for part in re.split(r"\bplus\b|;|,|\band\b", main, flags=re.IGNORECASE):
            part = part.strip()
            if part:
                labels.append(part)

    unique: List[str] = []
    seen = set()
    for label in labels:
        key = re.sub(r"\s+", " ", label.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(label)
    return unique


def resolve_source_refs(cited_paper: str, base_dir: str = ".") -> List[SourceRef]:
    refs: List[SourceRef] = []
    seen_paths = set()
    for label in requested_sources(cited_paper):
        path = resolve_pdf_path(label, base_dir)
        if not path:
            continue
        absolute = os.path.abspath(path) if os.path.isfile(path) else path
        if absolute in seen_paths:
            continue
        seen_paths.add(absolute)
        refs.append(SourceRef(raw=label, label=os.path.basename(absolute), path=absolute))
    return refs


def _image_area_ratio(page: "fitz.Page") -> float:
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        return 0.0
    page_area = abs(page.rect.width * page.rect.height)
    if page_area <= 0:
        return 0.0
    area = 0.0
    for info in infos:
        bbox = info.get("bbox")
        if not bbox:
            continue
        area += max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    return min(1.0, area / page_area)


def _native_text_inside_images(page: "fitz.Page") -> int:
    """Count native characters that sit inside raster image bounds.

    A page with a large raster figure and a one-line native caption is still a
    raster page: the caption does not contain the data in the figure. A vector
    figure has no raster bounds, so it remains native text.
    """

    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        return 0
    characters = 0
    for info in infos:
        bbox = info.get("bbox")
        if not bbox:
            continue
        try:
            clip = fitz.Rect(bbox)
            characters += len(page.get_text("text", clip=clip).strip())
        except Exception:
            continue
    return characters


def _sanitize_text(text: str) -> str:
    """Repair PDF glyph mappings that leak control characters.

    Some fonts use ``\\x01`` as a minus sign before digits and as a diacritic
    marker inside names. Only the numeric use is a sign. Remaining control
    characters carry no readable content.
    """

    text = re.sub(r"\x01(?=\d)", "-", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Join words split by a PDF line-break hyphen. This is a layout artifact,
    # not a missing character in the quote.
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", text)
    return text


def extract_pdf_pages(
    pdf_path: str,
    source_label: Optional[str] = None,
    dpi: int = 150,
) -> Dict[int, Dict[str, Any]]:
    """Extract pages as dictionaries with text, tables, images and provenance."""

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    label = source_label or os.path.basename(pdf_path)
    document = fitz.open(pdf_path)
    pages: Dict[int, Dict[str, Any]] = {}

    try:
        for index in range(len(document)):
            page_number = index + 1
            page = document[index]
            text = _sanitize_text(page.get_text("text") or "")
            stripped = text.strip()

            tables: List[str] = []
            try:
                for table in page.find_tables():
                    tables.append(_sanitize_text(table.to_markdown()))
            except Exception:
                tables = []

            image_ratio = _image_area_ratio(page)
            # A page with a large raster image and little native text inside the
            # image is a raster page even when a native caption sits below it.
            # Vector figures expose their labels in the text layer and do not
            # need a vision call.
            inside_text = _native_text_inside_images(page)
            needs_vision = (
                image_ratio >= 0.40
                and len(stripped) < 200
                and inside_text < 50
            )
            # A full-page raster with a thin native header has no readable
            # body text even though the header sits inside the image bounds.
            if image_ratio >= 0.70 and len(stripped) < 500:
                needs_vision = True

            image_bytes = None
            image_mime = "image/png"
            if needs_vision:
                pix = page.get_pixmap(dpi=dpi)
                image_bytes = pix.tobytes("png")
                image_mime = "image/png"

            if not stripped:
                provenance = "scanned"
            elif needs_vision:
                provenance = "mixed"
            else:
                provenance = "native"

            pages[page_number] = {
                "page_number": page_number,
                "source": label,
                "text": text,
                "tables": tables,
                "has_native_text": len(stripped) >= 80,
                "needs_vision": needs_vision,
                "image_bytes": image_bytes,
                "image_mime": image_mime,
                "provenance": provenance,
                "transcript": None,
            }
    finally:
        document.close()

    return pages
