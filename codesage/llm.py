"""Groq LLM wrapper - strict context enforcement for CodeSage."""
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL = "openai/gpt-oss-120b"
_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            try:
                import streamlit as st
                key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                pass
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY missing. Add it to .env (local) "
                "or Streamlit Cloud secrets (cloud)."
            )
        _client = Groq(api_key=key)
    return _client


def _chat(system: str, user: str, temperature: float = 0.2,
          max_tokens: int = 1500) -> str:
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ================================================================
# SYSTEM PROMPT
# ================================================================
BASE_SYSTEM = (
    "You are CodeSage, a Python code tutor. The user has provided Python "
    "code in this session. Every answer must be grounded in THAT code.\n\n"

    "=============== THE DECISION ===============\n"
    "Before answering, ask: 'Does this question relate to the code I was "
    "given in the context below?'\n"
    "  YES -> Answer with file + line citations and snippets from their code.\n"
    "  NO  -> Refuse with the exact message below. No exceptions.\n\n"

    "=============== WHAT COUNTS AS 'RELATED' ===============\n"
    "Answer if the question is about:\n"
    "- Any function, class, variable, line, or behavior in their code\n"
    "- Errors, bugs, security risks, style/naming issues in their code\n"
    "- Improvements, refactors, or tests for their code\n"
    "- Adding a feature to their code\n"
    "- A programming concept asked IN TERMS OF their code, i.e. the "
    "question mentions 'my code', 'this function', 'this variable', "
    "'here', 'instead of this', 'in this code', or references an "
    "identifier visible in the retrieved context.\n\n"

    "=============== WHAT TO REFUSE ===============\n"
    "Refuse if the question is:\n"
    "- Any non-programming topic: news, politics, sports, health, "
    "relationships, history, biology, medicine, law, geography, recipes, "
    "travel, entertainment, personal advice, essays, translations, jokes\n"
    "- A general programming concept with NO link to their code, e.g.:\n"
    "    * 'what is a static variable?' (no 'my code')\n"
    "    * 'what are Python data types?' (no code reference)\n"
    "    * 'explain decorators' (no code reference)\n"
    "    * 'what do you know about variable types?' (no code reference)\n"
    "  If the question doesn't reference their code (directly or by "
    "contextual phrase), REFUSE.\n\n"

    "=============== REFUSAL MESSAGE (exact) ===============\n"
    "\"I'm restricted to helping with your uploaded project. Please ask me "
    "something about your code - explaining a function, finding errors or "
    "security risks, suggesting improvements, or adding new features.\"\n\n"

    "=============== STYLE ===============\n"
    "- Cite file names and line numbers.\n"
    "- Use fenced code blocks for suggested code.\n"
    "- Match the project's existing style.\n"
    "- Never invent functions or files not in the context.\n"
    "- NEVER ask the user to paste code they already provided.\n\n"

    "Project style: {style}"
)
ERROR_AUDIT_ADDENDUM = (
    "\n\n=============== FULL ERROR & SECURITY AUDIT ===============\n"
    "The ENTIRE file was retrieved below - read it end to end before "
    "answering. Check EVERY line. Do NOT stop early.\n\n"

    "SCAN FOR (list each finding you can see):\n\n"

    "SECURITY (highest priority - start these with ⚠️ URGENT:):\n"
    "  - Hardcoded API keys (sk-, ghp_, AKIA, etc.), DB passwords, tokens\n"
    "  - SQL injection: string concatenation or f-strings in queries\n"
    "  - Command injection: os.system, subprocess with shell=True\n"
    "  - eval() / exec() on untrusted input\n"
    "  - pickle.loads / yaml.load on untrusted data\n"
    "  - Weak hashing (MD5, SHA1) for passwords\n"
    "  - Weak randomness (random module for tokens/secrets)\n"
    "  - Path traversal, insecure tempfile, disabled TLS verification\n\n"

    "LOGICAL:\n"
    "  - Division by zero, empty list indexing, off-by-one\n"
    "  - Missing None checks, missing returns on some branches\n"
    "  - Mutable default arguments (def f(items=[]))\n"
    "  - Infinite loops (while with no decrement/break)\n"
    "  - Unreachable code, wrong loop bounds\n\n"

    "SYNTAX / RUNTIME:\n"
    "  - Bare except: clauses, undefined names, wrong exception types\n"
    "  - Type mismatches, imports inside functions when they belong at top\n\n"

    "STYLE / NAMING:\n"
    "  - PEP 8 violations, non-snake_case functions, non-PascalCase classes\n"
    "  - Magic numbers, missing docstrings, missing type hints\n"
    "  - Dead code (unused functions, unused variables, unused imports)\n"
    "  - Functions longer than ~30 lines or with >5 args\n\n"

    "RESOURCE LEAKS:\n"
    "  - open() without 'with', unclosed DB connections\n\n"

    "OUTPUT - order by severity (critical first). Every finding uses:\n\n"
    "### <Short title>\n"
    "- **Severity**: critical | high | medium | low\n"
    "- **Type**: security | logical | syntax | style | naming | resource\n"
    "- **Where**: <file> - line <number>\n"
    "- **What**: <one sentence>\n"
    "- **Why it matters**: <one sentence>\n"
    "- **Fix**:\n"
    "    <short corrected snippet>\n\n"

    "RULES:\n"
    "- Scan the ENTIRE file top to bottom. If the file has 20+ issues, list "
    "all critical and high issues, then summarize the rest as bullets.\n"
    "- Never say 'no issues found' unless you genuinely scanned every line.\n"
    "- Never invent functions not in the retrieved code.\n"
    "- Hardcoded secrets are ALWAYS critical, no exceptions."
)
IMPROVEMENT_ADDENDUM = (
    "\n\n=============== IMPROVEMENT MODE ===============\n"
    "Give 3-6 concrete improvements:\n\n"
    "### Short title\n"
    "- Where: <file> - line <number>\n"
    "- Why: <one sentence>\n"
    "- How:\n"
    "    <suggested code>\n\n"
    "Never say 'I need more information' - use the retrieved code."
)

THINKING_ADDENDUM = (
    "\n\nTHINKING MODE: Analyze architecture, dependencies, patterns, "
    "edge cases. Structured answer with sections.\n"
)

FEATURE_ADDENDUM = (
    "\n\n=============== FEATURE-ADDITION MODE ===============\n"
    "1. Best place to add it (file + line).\n"
    "2. Exact code (project style).\n"
    "3. Integration steps.\n"
    "4. Short example.\n"
    "Format: ### Where to add - ### Code - ### Integration"
)


ADD_KEYWORDS = (
    "add", "implement", "create", "insert", "introduce",
    "how do i add", "how to add", "how can i add", "i want to add",
    "integrate", "extend",
)

ERROR_KEYWORDS = (
    "error", "errors", "bug", "bugs", "issue", "issues", "risk", "risks",
    "security", "vulnerab", "crash", "crashes", "exception",
    "wrong", "incorrect", "broken", "fail", "failure",
    "syntax error", "logical error", "runtime error",
    "audit", "review for", "find problems", "what is wrong",
    "what's wrong", "fix", "smell", "smells", "loophole",
    "loopholes", "exposed", "leak",
)

IMPROVE_KEYWORDS = (
    "improve", "improvement", "enhance", "enhancement", "refactor",
    "optimi", "suggestion", "suggest", "tips", "clean up", "cleanup",
    "what can be done", "what can i do", "make it better", "better",
    "replace", "instead of",
)


def detect_intent(question: str) -> str:
    q = question.lower().strip()
    if any(k in q for k in ERROR_KEYWORDS):
        return "error"
    if any(k in q for k in IMPROVE_KEYWORDS):
        return "improve"
    if any(k in q for k in ADD_KEYWORDS):
        return "add"
    if q.startswith(("what", "why", "explain", "how does", "how is", "describe")):
        return "explain"
    return "search"


def _build_context(hits: list) -> str:
    if not hits:
        return "(no matching context found)"
    blocks = []
    for h in hits:
        if h.get("kind") == "code":
            blocks.append(
                "--- [CODE] " + h["file"] + " :: " + h["name"]
                + " (lines " + str(h["line_start"]) + "-" + str(h["line_end"])
                + ") ---\n" + h["text"]
            )
        else:
            blocks.append(
                "--- [DOC] " + h["file"] + " page " + str(h.get("page", 1))
                + " ---\n" + h["text"]
            )
    return "\n\n".join(blocks)


def explain_function(func: dict, style: str) -> str:
    sys_p = BASE_SYSTEM.format(style=style)
    user = (
        "Explain this " + func["type"] + ".\n\n"
        "File: " + func["file"] + " (lines " + str(func["line_start"])
        + "-" + str(func["line_end"]) + ")\n"
        "Name: " + func["name"] + "\n\n"
        + func["text"] + "\n\n"
        "Cover: what it does, inputs, outputs, and any obvious issues."
    )
    return _chat(sys_p, user)


def answer_question(question: str, hits: list, style: str,
                    thinking: bool = False,
                    history: list = None) -> str:
    intent = detect_intent(question)

    system = BASE_SYSTEM.format(style=style)
    if intent == "error":
        system += ERROR_AUDIT_ADDENDUM
    if intent == "improve":
        system += IMPROVEMENT_ADDENDUM
    if intent == "add":
        system += FEATURE_ADDENDUM
    if thinking:
        system += THINKING_ADDENDUM

    context = _build_context(hits)

    history_block = ""
    if history:
        recent = history[-6:]
        history_block = "\n\nPrevious conversation:\n" + "\n".join(
            m["role"].upper() + ": " + m["content"][:300] for m in recent
        )

    user_prompt = (
        "User question: " + question + "\n\n"
        "Detected intent: " + intent + "\n\n"
        "Code from their project (this session):\n" + context
        + history_block + "\n\n"
        "===== DECISION PROCEDURE =====\n"
        "1. Does this question relate to the code above?\n"
        "   (a function, class, variable, line, error, security risk, "
        "improvement, feature, or programming concept asked in the "
        "context of that code)\n"
        "   -> YES: Answer with citations to the code above.\n\n"
        "2. Is it a general programming question NOT about the code above?\n"
        "   -> REFUSE with the exact message.\n\n"
        "3. Is it a non-programming question?\n"
        "   -> REFUSE with the exact message."
    )

    return _chat(
        system, user_prompt,
        temperature=0.2 if thinking else 0.1,
        max_tokens=3000 if intent == "error" else (2500 if thinking else 1800),
    )


def deep_analysis(question: str, hits: list, style: str, all_items: list) -> str:
    intent = detect_intent(question)
    system = BASE_SYSTEM.format(style=style) + THINKING_ADDENDUM
    if intent == "error":
        system += ERROR_AUDIT_ADDENDUM
    if intent == "improve":
        system += IMPROVEMENT_ADDENDUM

    file_tree = sorted(set(i["file"] for i in all_items))
    summary = (
        "Files in project (" + str(len(file_tree)) + "):\n"
        + "\n".join("- " + f for f in file_tree[:50])
    )
    context = _build_context(hits)

    user = (
        "Deep analysis request: " + question + "\n\n"
        "Project structure:\n" + summary + "\n\n"
        "Retrieved code:\n" + context + "\n\n"
        "Sections:\n"
        "### Overview\n"
        "### Architecture & Patterns\n"
        "### Security & Errors (critical first)\n"
        "### Improvements\n"
        "### Recommendations\n"
        "Answer only if the request relates to the code above; "
        "otherwise refuse."
    )
    return _chat(system, user, temperature=0.25, max_tokens=3500)


def explain_issue(issue: dict, style: str) -> str:
    sys_p = BASE_SYSTEM.format(style=style)
    user = (
        "Explain this static-analysis finding and give a one-line fix.\n\n"
        "Tool: " + issue["tool"] + "\n"
        "File: " + issue["file"] + " line " + str(issue["line"]) + "\n"
        "Code: " + issue["code"] + "\n"
        "Message: " + issue["message"] + "\n\n"
        "Format exactly:\nProblem: <one sentence>\nFix: <one-line suggestion>"
    )
    return _chat(sys_p, user, temperature=0.1)