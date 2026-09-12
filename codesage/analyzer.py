"""Static analysis via Ruff and Bandit."""
import subprocess
import json
import sys
from pathlib import Path


def run_ruff(project_root: Path) -> list[dict]:
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


def run_bandit(project_root: Path) -> list[dict]:
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


def analyze_project(project_root: Path) -> list[dict]:
    findings = run_ruff(project_root) + run_bandit(project_root)
    seen, unique = set(), []
    for f in findings:
        key = (f["file"], f["line"], f["message"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    return unique