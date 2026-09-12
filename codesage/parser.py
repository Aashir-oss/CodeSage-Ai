"""AST parsing for Python code + unified chunk schema."""
import ast
from pathlib import Path


def extract_items(file_path: Path, project_root: Path) -> list[dict]:
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
            })
    return results


def parse_project(py_files: list[Path], project_root: Path) -> list[dict]:
    items = []
    for f in py_files:
        items.extend(extract_items(f, project_root))
    return items


def build_embedding_text(item: dict) -> str:
    if item["kind"] == "code":
        parts = [f"[CODE] {item['type']}: {item['name']}",
                 f"File: {item['file']}"]
        if item["type"] == "function":
            parts.append(f"Args: {', '.join(item.get('args', []))}")
        if item.get("docstring"):
            parts.append(f"Docstring: {item['docstring']}")
        parts.append(f"Code:\n{item['text']}")
        return "\n".join(parts)
    return (f"[DOC] {item.get('name', 'chunk')} from {item['file']} "
            f"(page {item.get('page', 1)})\n{item['text']}")