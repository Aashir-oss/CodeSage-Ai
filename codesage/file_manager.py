"""ZIP extraction, file handling, and language detection (multi-language)."""
import re
import zipfile
import shutil
from pathlib import Path

IGNORE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    ".idea", ".vscode", "node_modules", ".pytest_cache",
    "build", "dist", ".mypy_cache", ".ruff_cache",
    "bin", "obj", "target", ".next", ".nuxt",
}

# ==============================================================
# SUPPORTED LANGUAGES
# ==============================================================
# Master list — every language the agent can process.
# Anything outside this dict is rejected with a friendly message.
EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java",
    ".cs": "csharp",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".hxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala",
    ".html": "html", ".htm": "html",
    ".xml": "xml",
    ".css": "css",
    ".scss": "css", ".sass": "css", ".less": "css",
    ".sql": "sql",
    ".sh": "bash", ".bash": "bash",
    ".ps1": "powershell",
    ".vue": "javascript",
    ".svelte": "javascript",
}

# Languages the LLM is told it can handle.
SUPPORTED_LANGUAGES = set(EXT_TO_LANG.values())

# Human-friendly list for error messages.
LANGUAGE_LIST_PRETTY = sorted(
    f"`{ext}` ({lang})" for ext, lang in EXT_TO_LANG.items()
)

CODE_EXTS = set(EXT_TO_LANG.keys())
DOC_EXTS   = {".pdf", ".docx", ".txt", ".md", ".rst"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


# ==============================================================
# LANGUAGE DETECTION
# ==============================================================
def language_for(path: Path) -> str:
    """Return the language for a file, or 'unknown' if unsupported."""
    return EXT_TO_LANG.get(path.suffix.lower(), "unknown")


def is_supported_file(path: Path) -> bool:
    """True if the file's extension maps to a supported language."""
    return path.suffix.lower() in EXT_TO_LANG


def extension_for_language(lang: str) -> str:
    """Reverse map: language → canonical file extension."""
    for ext, l in EXT_TO_LANG.items():
        if l == lang:
            return ext
    return ".txt"


def detect_language_from_text(text: str) -> str:
    """Detect the language of pasted code. Returns a language name
    from EXT_TO_LANG.values(), or 'unknown' if nothing matches."""
    snippet = text[:3000]

    checks = [
        (r"<!DOCTYPE\s+html|<html[\s>]|<head[\s>]|<body[\s>]", "html"),
        (r"<\?xml\s", "xml"),
        (r"^\s*[.#][\w-]+\s*\{[^}]*\}", "css"),
        (r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE\s+TABLE)\b[\s\S]*\b(FROM|INTO|SET|VALUES)\b",
         "sql"),
        (r"^\s*(?:import\s+\w+|from\s+\w+\s+import|def\s+\w+\s*\(|class\s+\w+)",
         "python"),
        (r"<\?php", "php"),
        (r"#!/bin/(?:ba)?sh|^\s*echo\s+", "bash"),
        (r"\bparam\s*\(|\bGet-ChildItem\b|\bWrite-Host\b", "powershell"),
        (r"^\s*(?:pub\s+)?(?:fn\s+\w+|let\s+mut\s|use\s+crate)", "rust"),
        (r"^\s*package\s+\w+|^\s*func\s+\w+", "go"),
        (r"\busing\s+System;|namespace\s+\w+|Console\.WriteLine", "csharp"),
        (r"\bpublic\s+class\s+\w+|System\.out\.print|private\s+static\s+void",
         "java"),
        (r"#include\s*<[\w./]+>|\bint\s+main\s*\(", "cpp"),
        (r":\s*(?:string|number|boolean|any|void)\b|interface\s+\w+",
         "typescript"),
        (r"\b(?:function|const|let|var)\s+\w+|=>\s*\{|console\.log\(",
         "javascript"),
        (r"^\s*def\s+\w+[\s\S]*?\bend\b|require\s+'", "ruby"),
        (r"\bfunc\s+\w+|@IBOutlet|override\s+func", "swift"),
        (r"\bfun\s+\w+|val\s+\w+|var\s+\w+:\s*\w+", "kotlin"),
        (r"\bobject\s+\w+\s+extends\s+|\bdef\s+\w+\s*\(", "scala"),
    ]

    for pattern, lang in checks:
        if re.search(pattern, snippet, re.MULTILINE | re.IGNORECASE):
            # Disambiguate Java vs C#
            if lang == "java" and re.search(r"\busing\s+System|namespace\s+", snippet):
                return "csharp"
            return lang

    return "unknown"


# ==============================================================
# ZIP EXTRACTION
# ==============================================================
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


# ==============================================================
# FILE SCANNING
# ==============================================================
def find_python_files(project_root: Path) -> list:
    """Backward-compat helper — returns all code files."""
    return find_all_supported_files(project_root)["code"]


def find_all_supported_files(project_root: Path) -> dict:
    """Return {'code': [...], 'docs': [...], 'images': [...]} for the project."""
    code, docs, images, unsupported = [], [], [], []
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
        elif ext:
            unsupported.append(path)
    return {
        "code": code,
        "docs": docs,
        "images": images,
        "unsupported": unsupported,
    }


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