"""Detect project coding conventions."""
import re
from collections import Counter
from pathlib import Path


def learn_style(project_root: Path, py_files: list[Path], items: list[dict]) -> dict:
    code_chunks = []
    for f in py_files:
        try:
            code_chunks.append(f.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass
    code = "\n".join(code_chunks)

    func_names = [i["name"] for i in items if i["type"] == "function"]
    snake = sum(1 for n in func_names if re.fullmatch(r"[a-z_][a-z0-9_]*", n))
    camel = sum(1 for n in func_names if re.fullmatch(r"[a-z]+([A-Z][a-z0-9]*)+", n))
    total = max(len(func_names), 1)
    if snake / total > 0.6:
        naming = "snake_case"
    elif camel / total > 0.6:
        naming = "camelCase"
    else:
        naming = "mixed"

    raises = Counter(re.findall(r"raise\s+(\w+)", code))
    error_pattern = "try/except" if "try:" in code else "no try/except"
    common_raises = [f"{k} ({v}x)" for k, v in raises.most_common(3)]

    documented = sum(1 for i in items if i.get("docstring"))
    doc_ratio = documented / max(len(items), 1)

    imports = Counter(re.findall(r"^(?:from|import)\s+([\w\.]+)", code, re.MULTILINE))
    top_imports = [k for k, _ in imports.most_common(5)]

    summary = (
        f"Naming: uses {naming} for functions. "
        f"Error handling: {error_pattern}"
        + (f", commonly raises {', '.join(common_raises)}. " if common_raises else ". ")
        + f"Documentation: {int(doc_ratio * 100)}% of items have docstrings."
        + (f" Top imports: {', '.join(top_imports)}." if top_imports else "")
    )

    return {
        "naming": naming,
        "error_pattern": error_pattern,
        "common_raises": common_raises,
        "doc_ratio": round(doc_ratio, 2),
        "top_imports": top_imports,
        "summary": summary,
    }