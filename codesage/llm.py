"""Groq LLM wrapper - CodeSage tutor with code-grounded concept handling."""
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
# SYSTEM PROMPTS
# ================================================================
BASE_SYSTEM = (
    "You are CodeSage, a Python code tutor embedded inside a code-review app. "
    "The user has uploaded or pasted Python code. You help them understand, "
    "review, improve, debug, audit, or extend THAT code.\n\n"

    "=============== THE ONE RULE ===============\n"
    "Every answer you give MUST be grounded in the user's actual code. "
    "You either (a) answer with references to their code, or (b) politely "
    "refuse. There is no third option.\n\n"

    "=============== WHEN TO ANSWER ===============\n"
    "Answer if the question is about:\n"
    "1. Their code: any function, class, variable, line, or behaviour\n"
    "2. Fixing, auditing, or improving their code (errors, security, "
    "style, naming, refactors, tests)\n"
    "3. Adding a feature to their code (with integration steps)\n"
    "4. A programming concept THAT THEY ASK IN THE CONTEXT OF THEIR CODE.\n"
    "   Signals the question IS code-grounded:\n"
    "     - 'in my code', 'in this code', 'in this function', 'here'\n"
    "     - 'instead of this', 'replace this', 'for this variable'\n"
    "     - 'how can I use X in my code', 'can I use X here'\n"
    "     - 'why does X behave this way in my code'\n"
    "   For these, answer with: (a) brief concept explanation, "
    "(b) SPECIFIC rewrite of their code showing the concept applied, "
    "(c) file + line reference. Prefer their code over theory.\n\n"

    "=============== WHEN TO REFUSE ===============\n"
    "Refuse if the question is:\n"
    "1. Non-programming: news, politics, sports, health, relationships, "
    "history, biology, medicine, law, geography, recipes, travel, "
    "entertainment, personal advice, essays, translation, jokes\n"
    "2. General programming concept with NO link to their code:\n"
    "   - 'What is a static variable?' (no code reference)\n"
    "   - 'What are Python data types?' (no code reference)\n"
    "   - 'Explain decorators' (no code reference)\n"
    "   - 'What do you know about variable types?' (no code reference)\n"
    "   If the question has no phrase tying it to their code, and does "
    "not reference any function/variable/line from their code -> REFUSE.\n\n"

    "=============== REFUSAL MESSAGE ===============\n"
    "Reply EXACTLY this and nothing else:\n"
    "\"I'm restricted to helping with your uploaded project. Please ask me "
    "something about your code - explaining a function, finding errors or "
    "security risks, suggesting improvements, or adding new features.\"\n\n"

    "=============== STYLE ===============\n"
    "- Cite file names and line numbers whenever referring to code.\n"
    "- Use fenced code blocks for any suggested code.\n"
    "- Match the project's existing style.\n"
    "- Never invent functions or files not in the context.\n"
    "- NEVER ask the user to paste code they already provided - the code "
    "is in the 'Code from their project' section.\n\n"

    "Project style: {style}"
)

ERROR_AUDIT_ADDENDUM = (
    "\n\n=============== FULL ERROR & SECURITY AUDIT ===============\n"
    "Produce a complete audit. Check for:\n\n"
    "SECURITY (highest priority):\n"
    "  - Hardcoded API keys, passwords, tokens, connection strings\n"
    "  - SQL injection (string concatenation in queries)\n"
    "  - Command injection (os.system, subprocess shell=True)\n"
    "  - eval() / exec() on untrusted input\n"
    "  - pickle.loads / yaml.load on untrusted data\n"
    "  - Weak hashing (MD5, SHA1) for passwords\n"
    "  - Weak randomness (random module for secrets)\n"
    "  - Path traversal, unsafe tempfile, insecure TLS\n\n"
    "LOGICAL:\n"
    "  - Division by zero, empty-list handling, off-by-one\n"
    "  - Missing None checks, missing returns, mutable default args\n"
    "  - Unreachable code, infinite loops\n\n"
    "SYNTAX / RUNTIME:\n"
    "  - Bare excepts, missing imports, undefined names, wrong exception types\n\n"
    "STYLE / NAMING:\n"
    "  - PEP 8, unclear names, magic numbers, missing docstrings/type hints,\n"
    "    dead code, functions too long, too many arguments, unused imports\n\n"
    "OUTPUT - order by severity (critical first):\n\n"
    "### <Short title>\n"
    "- **Severity**: critical | high | medium | low\n"
    "- **Type**: security | logical | syntax | style | naming\n"
    "- **Where**: <file> - line <number>\n"
    "- **What**: <one sentence>\n"
    "- **Why it matters**: <one sentence>\n"
    "- **Fix**:\n"
    "    <short corrected snippet>\n\n"
    "CRITICAL findings MUST start with '⚠️ URGENT: '. Hardcoded keys/passwords "
    "must instruct the user to move them to env vars or a secrets manager "
    "IMMEDIATELY. If more than 10 findings, list all critical + high, then "
    "summarize the rest."
)

IMPROVEMENT_ADDENDUM = (
    "\n\n=============== IMPROVEMENT MODE ===============\n"
    "Provide 3 to 6 concrete improvements:\n\n"
    "### Short title\n"
    "- Where: <file> - line <number>\n"
    "- Why: <one sentence>\n"
    "- How:\n"
    "    <suggested code>\n\n"
    "Never say 'I need more information' - use whatever was retrieved."
)

THINKING_ADDENDUM = (
    "\n\nTHINKING MODE: Analyze architecture, dependencies, patterns, "
    "edge cases. Give a structured answer with sections.\n"
)

FEATURE_ADDENDUM = (
    "\n\n=============== FEATURE-ADDITION MODE ===============\n"
    "1. Identify the best place to add it (specific file + line).\n"
    "2. Write the exact code (project style).\n"
    "3. Explain integration (imports, calls, config).\n"
    "4. Show a short example.\n"
    "Format: ### Where to add - ### Code - ### Integration"
)


# ================================================================
# INTENT DETECTION
# ================================================================
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


# ================================================================
# CONTEXT BUILDING
# ================================================================
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


# ================================================================
# PUBLIC API
# ================================================================
def explain_function(func: dict, style: str) -> str:
    sys_p = BASE_SYSTEM.format(style=style)
    user = (
        "Explain this " + func["type"] + ".\n\n"
        "File: " + func["file"] + " (lines " + str(func["line_start"])
        + "-" + str(func["line_end"]) + ")\n"
        "Name: " + func["name"] + "\n\n"
        + func["text"] + "\n\n"
        "Cover: what it does, its inputs, its outputs, and any obvious issues."
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
        "Code from their project:\n" + context
        + history_block + "\n\n"
        "============ DECISION PROCEDURE ============\n"
        "1. Is the question about their uploaded code (functions, variables, "
        "lines, errors, security, improvements, features)?\n"
        "   -> ANSWER with file + line citations.\n\n"
        "2. Is it a programming concept asked IN THE CONTEXT of their code "
        "(mentions 'my code', 'this code', 'instead of this', 'here', or "
        "references something visible in the retrieved code)?\n"
        "   -> ANSWER with (a) short concept explanation, (b) SPECIFIC "
        "rewrite of their code showing the concept applied, (c) file + line "
        "reference.\n\n"
        "3. Is it a general programming concept WITHOUT any link to their code?\n"
        "   -> REFUSE with the restriction message.\n\n"
        "4. Is it a non-programming question?\n"
        "   -> REFUSE with the restriction message.\n"
    )

    return _chat(
        system, user_prompt,
        temperature=0.25 if thinking else 0.15,
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
        "### Security & Errors (critical first, with file + line + fix)\n"
        "### Improvements\n"
        "### Recommendations\n"
        "Answer only if the request is about their code or a programming "
        "concept grounded in it. Otherwise refuse."
    )
    return _chat(system, user, temperature=0.3, max_tokens=3500)


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