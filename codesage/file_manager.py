"""ZIP extraction + individual file handling."""
import zipfile
import shutil
from pathlib import Path

IGNORE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    ".idea", ".vscode", "node_modules", ".pytest_cache",
    "build", "dist", ".mypy_cache", ".ruff_cache",
}

CODE_EXTS  = {".py"}
DOC_EXTS   = {".pdf", ".docx", ".txt", ".md", ".rst"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def extract_zip(zip_path: str, dest_dir: str) -> Path:
    dest = Path(dest_dir)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    children = [p for p in dest.iterdir() if p.is_dir()]
    files = [p for p in dest.iterdir() if p.is_file()]
    if len(children) == 1 and not files:
        return children[0]
    return dest


def find_python_files(project_root: Path) -> list[Path]:
    py_files = []
    for path in project_root.rglob("*.py"):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        py_files.append(path)
    return py_files


def find_all_supported_files(project_root: Path) -> dict:
    code, docs, images = [], [], []
    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        ext = path.suffix.lower()
        if ext in CODE_EXTS:
            code.append(path)
        elif ext in DOC_EXTS:
            docs.append(path)
        elif ext in IMAGE_EXTS:
            images.append(path)
    return {"code": code, "docs": docs, "images": images}


def save_uploaded_file(uploaded_file, dest_dir: str) -> Path:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / uploaded_file.name
    path.write_bytes(uploaded_file.getbuffer())
    return path


def scan_summary(project_root: Path) -> dict:
    all_files = find_all_supported_files(project_root)
    total_loc = 0
    for f in all_files["code"]:
        try:
            total_loc += len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
        except Exception:
            pass
    return {
        "project_root": str(project_root),
        "file_count": len(all_files["code"]),
        "doc_count": len(all_files["docs"]),
        "image_count": len(all_files["images"]),
        "total_loc": total_loc,
    }