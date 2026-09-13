"""Multi-language parser using regex heuristics."""
import ast
import re
from pathlib import Path

from codesage.file_manager import language_for, EXT_TO_LANG


# ------------------------------------------------------------------
# Regex patterns for common function declarations per language
# Each pattern captures the function name in group 1
# ------------------------------------------------------------------
PATTERNS = {
    "python": [
        re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\("),
        re.compile(r"^\s*class\s+(\w+)\s*[\(:]"),
    ],
    "javascript": [
        re.compile(r"^\s*(?:async\s+)?function\s+(\w+)\s*\("),
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\()"),
        re.compile(r"^\s*(?:export\s+)?class\s+(\w+)"),
        re.compile(r"^\s{2,}(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{"),
    ],
    "typescript": [
        re.compile(r"^\s*(?:async\s+)?function\s+(\w+)\s*\("),
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\()"),
        re.compile(r"^\s*(?:export\s+)?class\s+(\w+)"),
        re.compile(r"^\s*(?:public|private|protected|static)?\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*(?::\s*\w+)?\s*\{"),
        re.compile(r"^\s*(?:export\s+)?interface\s+(\w+)"),
    ],
    "java": [
        re.compile(r"^\s*(?:public|private|protected|static|final|synchronized|\s)+\w[\w<>\[\],\s]*\s+(\w+)\s*\("),
        re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?class\s+(\w+)"),
        re.compile(r"^\s*(?:public|private|protected)?\s*interface\s+(\w+)"),
    ],
    "csharp": [
        re.compile(r"^\s*(?:public|private|protected|internal|static|virtual|override|async|\s)+[\w<>\[\],\s]+\s+(\w+)\s*\("),
        re.compile(r"^\s*(?:public|private|internal)?\s*(?:sealed\s+|abstract\s+|static\s+)?class\s+(\w+)"),
    ],
    "cpp": [
        re.compile(r"^\s*(?:[\w:<>~*&]+\s+)+(\w+)\s*\([^;]*\)\s*(?:const)?\s*\{"),
        re.compile(r"^\s*(?:class|struct)\s+(\w+)"),
    ],
    "c": [
        re.compile(r"^\s*(?:[\w*]+\s+)+(\w+)\s*\([^;]*\)\s*\{"),
        re.compile(r"^\s*(?:struct|union|enum)\s+(\w+)"),
    ],
    "go": [
        re.compile(r"^\s*func\s+(?:\([^)]*\)\s+)?(\w+)\s*\("),
        re.compile(r"^\s*type\s+(\w+)\s+(?:struct|interface)"),
    ],
    "rust": [
        re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)"),
        re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait|impl)\s+(\w+)"),
    ],
    "ruby": [
        re.compile(r"^\s*def\s+(\w+)"),
        re.compile(r"^\s*class\s+(\w+)"),
        re.compile(r"^\s*module\s+(\w+)"),
    ],
    "php": [
        re.compile(r"^\s*(?:public|private|protected|static)?\s*function\s+(\w+)"),
        re.compile(r"^\s*(?:abstract\s+|final\s+)?class\s+(\w+)"),
    ],
    "swift": [
        re.compile(r"^\s*(?:public|private|internal|fileprivate)?\s*func\s+(\w+)"),
        re.compile(r"^\s*(?:public|private|internal)?\s*class\s+(\w+)"),
        re.compile(r"^\s*(?:public|private|internal)?\s*struct\s+(\w+)"),
    ],
    "kotlin": [
        re.compile(r"^\s*(?:public|private|internal|protected)?\s*fun\s+(\w+)"),
        re.compile(r"^\s*(?:public|private|internal)?\s*class\s+(\w+)"),
    ],
    "scala": [
        re.compile(r"^\s*(?:def|val|var)\s+(\w+)"),
        re.compile(r"^\s*(?:class|object|trait)\s+(\w+)"),
    ],
    "sql": [
        re.compile(r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE|TABLE|VIEW)\s+(\w+)", re.IGNORECASE),
    ],
    "bash": [
        re.compile(r"^\s*(?:function\s+)?(\w+)\s*\(\)\s*\{"),
    ],
    "powershell": [
        re.compile(r"^\s*function\s+(\w+)"),
    ],
    "html": [
        # Not much to parse; skip
    ],
    "css": [
        # Not code functions; skip
    ],
    "vue": [
        # treated as javascript mostly
    ],
    "svelte": [
        # treated as javascript mostly
    ],
    "scss": [re.compile(r"^\s*@mixin\s+(\w+)")],
    "sass": [],
    "less": [],
}


def _extract_blocks(source: str, patterns: list) -> list:
    """Find function/class blocks using regex + line-based heuristic."""
    lines = source.splitlines()
    results = []
    n = len(lines)

    for i, line in enumerate(lines):
        for pat in patterns:
            m = pat.match(line)
            if not m:
                continue
            name = m.group(1)
            start = i

            # Find the end of the block by tracking brace / indent balance
            end = _find_block_end(lines, start)
            text = "\n".join(lines[start:end + 1])
            results.append({
                "name": name,
                "line_start": start + 1,
                "line_end": end + 1,
                "text": text,
            })
            break  # don't match twice on the same line

    return results


def _find_block_end(lines: list, start: int) -> int:
    """Find last line of a code block starting at `start`."""
    first = lines[start]
    # Python-style: indentation-based block
    if first.rstrip().endswith(":"):
        base_indent = len(first) - len(first.lstrip())
        i = start + 1
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent:
                return i - 1
            i += 1
        return len(lines) - 1

    # Brace-based block
    if "{" in first:
        depth = 0
        i = start
        while i < len(lines):
            depth += lines[i].count("{") - lines[i].count("}")
            if depth <= 0 and i > start:
                return i
            i += 1
        return len(lines) - 1

    # Arrow function / expression — single line
    return start


def extract_items(file_path: Path, project_root: Path) -> list:
    ext = file_path.suffix.lower()
    lang = EXT_TO_LANG.get(ext)
    if not lang:
        return []

    # Python: prefer real AST for accuracy
    if lang == "python":
        return _extract_python(file_path, project_root)

    # Everything else: regex-based
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    patterns = PATTERNS.get(lang, [])
    if not patterns:
        return []

    try:
        rel_path = str(file_path.relative_to(project_root))
    except Exception:
        rel_path = file_path.name

    items = []
    for block in _extract_blocks(source, patterns):
        items.append({
            "kind": "code",
            "type": "function",
            "name": block["name"],
            "file": rel_path,
            "line_start": block["line_start"],
            "line_end": block["line_end"],
            "text": block["text"],
            "docstring": "",
            "language": lang,
        })
    return items


def _extract_python(file_path: Path, project_root: Path) -> list:
    """Python AST-based extraction (kept for accuracy)."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except Exception:
        return []

    try:
        rel_path = str(file_path.relative_to(project_root))
    except Exception:
        rel_path = file_path.name

    lines = source.splitlines()
    results = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = node.lineno, node.end_lineno or node.lineno
            results.append({
                "kind": "code", "type": "function", "name": node.name,
                "file": rel_path, "line_start": start, "line_end": end,
                "args": [a.arg for a in node.args.args],
                "docstring": ast.get_docstring(node) or "",
                "text": "\n".join(lines[start - 1:end]),
                "language": "python",
            })
        elif isinstance(node, ast.ClassDef):
            start, end = node.lineno, node.end_lineno or node.lineno
            methods = [n.name for n in node.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            results.append({
                "kind": "code", "type": "class", "name": node.name,
                "file": rel_path, "line_start": start, "line_end": end,
                "methods": methods,
                "docstring": ast.get_docstring(node) or "",
                "text": "\n".join(lines[start - 1:end]),
                "language": "python",
            })
    return results


def parse_project(code_files: list, project_root: Path) -> list:
    items = []
    for f in code_files:
        items.extend(extract_items(f, project_root))
    return items


def build_embedding_text(item: dict) -> str:
    lang = item.get("language", "python")
    if item.get("kind") == "code":
        parts = [
            f"[{lang.upper()}] {item['type']}: {item['name']}",
            f"File: {item['file']}",
        ]
        if item["type"] == "function" and item.get("args"):
            parts.append(f"Args: {', '.join(item['args'])}")
        if item.get("docstring"):
            parts.append(f"Docstring: {item['docstring']}")
        parts.append(f"Code:\n{item['text']}")
        return "\n".join(parts)
    return (f"[DOC] {item.get('name', 'chunk')} from {item['file']} "
            f"(page {item.get('page', 1)})\n{item['text']}")