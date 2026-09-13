"""Static analysis. Ruff + Bandit for Python; LLM handles other languages."""
import subprocess
import json
import sys
from pathlib import Path

from codesage.file_manager import CODE_EXTS


def run_ruff(project_root: Path) -> list:
    try:
        r = subprocess.run(
            [sys.executable, "-m", "ruff", "check",
             "--output-format=json", str(project_root)],
            capture_output=True, text=True, timeout=60,
        )
        raw = json.loads(r.stdout or "[]")
    except Exception:
        return []
    return [{
        "tool": "ruff",
        "file": item.get("filename", ""),
        "line": item.get("location", {}).get("row", 0),
        "code": item.get("code", ""),
        "message": item.get("message", ""),
        "severity": "medium",
    } for item in raw]


def run_bandit(project_root: Path) -> list:
    try:
        r = subprocess.run(
            [sys.executable, "-m", "bandit",
             "-r", str(project_root), "-f", "json", "-q"],
            capture_output=True, text=True, timeout=120,
        )
        raw = json.loads(r.stdout or "{}")
    except Exception:
        return []
    return [{
        "tool": "bandit",
        "file": item.get("filename", ""),
        "line": item.get("line_number", 0),
        "code": item.get("test_id", ""),
        "message": item.get("issue_text", ""),
        "severity": item.get("issue_severity", "LOW").lower(),
    } for item in raw.get("results", [])]


def analyze_project(project_root: Path) -> list:
    """Run Python linters only if Python files exist. Other languages are
    reviewed by the LLM through the audit prompt."""
    findings = []
    py_files = [p for p in project_root.rglob("*.py")
                if not any(part.startswith(".") for part in p.parts)]

    if py_files:
        findings = run_ruff(project_root) + run_bandit(project_root)

    # Deduplicate
    seen, unique = set(), []
    for f in findings:
        key = (f["file"], f["line"], f["message"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    return unique