"""Document loaders for PDF, DOCX, TXT, MD, and images."""
from pathlib import Path


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def load_pdf(path: Path) -> list[dict]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    try:
        reader = PdfReader(str(path))
    except Exception:
        return []
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        for i, chunk in enumerate(_chunk_text(text)):
            chunks.append({
                "kind": "doc", "type": "pdf_page",
                "name": f"{path.name} p{page_num} c{i+1}",
                "file": path.name, "page": page_num, "text": chunk,
            })
    return chunks


def load_docx(path: Path) -> list[dict]:
    try:
        from docx import Document
    except ImportError:
        return []
    try:
        doc = Document(str(path))
    except Exception:
        return []
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    chunks = []
    for i, chunk in enumerate(_chunk_text(full_text)):
        chunks.append({
            "kind": "doc", "type": "docx",
            "name": f"{path.name} chunk {i+1}",
            "file": path.name, "page": i + 1, "text": chunk,
        })
    return chunks


def load_text(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    chunks = []
    for i, chunk in enumerate(_chunk_text(text)):
        chunks.append({
            "kind": "doc", "type": "text",
            "name": f"{path.name} chunk {i+1}",
            "file": path.name, "page": i + 1, "text": chunk,
        })
    return chunks


def load_image(path: Path) -> list[dict]:
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return []
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
    except Exception:
        return []
    if not text.strip():
        return []
    chunks = []
    for i, chunk in enumerate(_chunk_text(text)):
        chunks.append({
            "kind": "doc", "type": "image_ocr",
            "name": f"{path.name} chunk {i+1}",
            "file": path.name, "page": i + 1, "text": chunk,
        })
    return chunks


def load_document(path: Path) -> list[dict]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext in {".txt", ".md", ".rst"}:
        return load_text(path)
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
        return load_image(path)
    return []


def load_all_documents(doc_paths: list[Path], image_paths: list[Path]) -> list[dict]:
    items = []
    for p in doc_paths:
        items.extend(load_document(p))
    for p in image_paths:
        items.extend(load_document(p))
    return items