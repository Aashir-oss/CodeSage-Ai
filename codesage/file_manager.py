"""ZIP extraction + individual file handling (multi-language)."""
import zipfile
import shutil
from pathlib import Path

IGNORE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    ".idea", ".vscode", "node_modules", ".pytest_cache",
    "build", "dist", ".mypy_cache", ".ruff_cache",
    "bin", "obj", "target", ".next", ".nuxt",
}

# Languages we support for code parsing
CODE_EXTS = {
    ".py",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java",
    ".cs",
    ".cpp", ".cc", ".cxx", ".hpp", ".h", ".hxx",
    ".c",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift", ".kt", ".kts",
    ".scala",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".vue", ".svelte",
    ".sql",
    ".sh", ".bash", ".ps1",
}

DOC_EXTS   = {".pdf", ".docx", ".txt", ".md", ".rst"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

# Extension → language name (used for prompts and metadata)
EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".mjs": "javascript", ".cjs": "javascript",
    ".java": "java",
    ".cs": "csharp",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".h": "cpp", ".hxx": "cpp",
    ".c": "c",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
    ".vue": "vue", ".svelte": "svelte",
    ".sql": "sql",
    ".sh": "bash", ".bash": "bash", ".ps1": "powershell",
}


def language_for(path: Path) -> str:
    return EXT_TO_LANG.get(path.suffix.lower(), "unknown")


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


def find_python_files(project_root: Path) -> list:
    """Kept for backward compatibility — returns all code files now."""
    return find_all_supported_files(project_root)["code"]


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