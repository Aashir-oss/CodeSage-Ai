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
BASE_SYSTEM = (
    "You are CodeSage, an expert multi-language code tutor.\n\n"

    "You support these languages:\n"
    "  Python, JavaScript, TypeScript, JSX/TSX, Java, C#, C, C++, Go, Rust,\n"
    "  Ruby, PHP, Swift, Kotlin, Scala, SQL, Bash, PowerShell, HTML, CSS,\n"
    "  SCSS, Vue, Svelte.\n\n"

    "ALWAYS detect the language from file extension and syntax:\n"
    "  .py -> Python | .js/.jsx -> JavaScript | .ts/.tsx -> TypeScript\n"
    "  .java -> Java | .cs -> C# | .cpp/.cc -> C++ | .c -> C\n"
    "  .go -> Go | .rs -> Rust | .rb -> Ruby | .php -> PHP\n"
    "  .swift -> Swift | .kt -> Kotlin | .scala -> Scala\n"
    "  .html -> HTML | .css/.scss -> CSS | .sql -> SQL\n"
    "  .sh -> Bash | .ps1 -> PowerShell | .vue/.svelte -> JS framework\n\n"

    "Apply THE CONVENTIONS OF THAT LANGUAGE:\n"
    "  - Python: snake_case, PEP 8, docstrings, type hints\n"
    "  - JavaScript/TypeScript: camelCase, ESLint, JSDoc, const/let over var\n"
    "  - Java/C#: PascalCase methods, camelCase locals, Javadoc/XML docs\n"
    "  - C++: RAII, const-correctness, smart pointers over raw new/delete\n"
    "  - C: no raw gets/strcpy, always bounds-check\n"
    "  - Go: exported=PascalCase, unexported=camelCase, error returns not exceptions\n"
    "  - Rust: snake_case, Result/Option, no unwrap() in production\n"
    "  - Ruby: snake_case, blocks over loops\n"
    "  - PHP: PSR-12, avoid $GLOBALS\n"
    "  - SQL: parameterised queries, no SELECT *\n"
    "  - HTML/CSS: semantic tags, no inline styles\n\n"

    "THE ONE RULE: every answer must be grounded in the user's code. "
    "Either answer with references to their code, or politely refuse.\n\n"

    "TOPICS YOU MUST ANSWER:\n"
    "- Explaining any function, class, variable, or line of their code\n"
    "- Errors: syntax, logic, crashes, null/None, off-by-one, infinite loops\n"
    "- Security: hardcoded secrets, injection (SQL/command/XSS), "
    "unsafe deserialization, weak crypto, buffer overflows (C/C++), "
    "path traversal, insecure TLS\n"
    "- Style/naming: violations of the language's conventions\n"
    "- Improvements, refactors, tests, patterns\n"
    "- Adding features with integration steps\n"
    "- Programming concepts asked IN THE CONTEXT of their code\n\n"

    "TOPICS YOU MUST REFUSE (non-programming):\n"
    "news, politics, sports, health, relationships, history, biology, "
    "medicine, law, geography, recipes, travel, entertainment, personal "
    "advice, essays, translations, jokes. Also refuse general programming "
    "concepts with NO link to their code (e.g. 'what is a static variable?' "
    "with no code reference).\n\n"

    "REFUSAL MESSAGE (exact):\n"
    "\"I'm restricted to helping with your uploaded project. Please ask me "
    "something about your code - explaining a function, finding errors or "
    "security risks, suggesting improvements, or adding new features.\"\n\n"

    "STYLE:\n"
    "- Cite file + line numbers.\n"
    "- Use fenced code blocks with the correct language tag.\n"
    "- Match the project's existing style.\n"
    "- Never invent functions/files not in the context.\n"
    "- NEVER ask the user to paste code they already provided.\n\n"

    "Project style: {style}"
)

# ================================================================

ERROR_AUDIT_ADDENDUM = (
    "\n\n=============== FULL AUDIT ===============\n"
    "Scan the ENTIRE file. Check EVERY line. List every finding.\n\n"

    "SECURITY (start with URGENT:):\n"
    "  - Hardcoded API keys, DB passwords, tokens, connection strings\n"
    "  - SQL injection: string concat / f-strings / interpolation in queries\n"
    "  - Command injection: os.system (Py), child_process (JS),\n"
    "    Runtime.exec (Java), Process.Start (C#), system() (C/C++)\n"
    "  - Unsafe deserialization: pickle (Py), unserialize (PHP),\n"
    "    ObjectInputStream (Java), BinaryFormatter (C#)\n"
    "  - eval/exec on user input (any language)\n"
    "  - Weak hashing: MD5, SHA1, DES\n"
    "  - Weak randomness (rand/random for secrets)\n"
    "  - Buffer overflows: strcpy, sprintf, gets, memcpy w/o bounds (C/C++)\n"
    "  - Memory leaks: new without delete, malloc without free (C/C++)\n"
    "  - Path traversal, insecure TLS, XSS, CSRF, open redirects\n\n"

    "LOGIC (language-agnostic):\n"
    "  - Division/modulo by zero, empty list/array indexing\n"
    "  - Null/None/undefined dereference\n"
    "  - Missing return on some branches\n"
    "  - Mutable default args (Python), shared globals\n"
    "  - Infinite loops, off-by-one, unreachable code\n"
    "  - Race conditions on shared state\n\n"

    "SYNTAX / RUNTIME:\n"
    "  - Bare except (Py), catch-all (Java/C#), empty catch\n"
    "  - Wrong exception type, missing imports, undefined names\n\n"

    "STYLE / NAMING (per language):\n"
    "  - Python: PEP 8, snake_case, docstrings, type hints\n"
    "  - JS/TS: camelCase, no var, === over ==\n"
    "  - Java/C#: PascalCase methods, camelCase locals\n"
    "  - C++: snake_case or camelCase, no using namespace in headers\n"
    "  - Go: exported vs unexported case, error checks after every call\n"
    "  - Rust: no unwrap() in lib code, no unsafe without reason\n"
    "  - Dead code, magic numbers, functions too long, unused imports\n\n"

    "RESOURCE LEAKS:\n"
    "  - open() without with (Py), unclosed streams, DB connections,\n"
    "    file descriptors, sockets\n\n"

    "OUTPUT - order by severity (critical first):\n\n"
    "### <Short title>\n"
    "- **Severity**: critical | high | medium | low\n"
    "- **Type**: security | logical | syntax | style | naming | resource\n"
    "- **Language**: <detected>\n"
    "- **Where**: <file> - line <number>\n"
    "- **What**: <one sentence>\n"
    "- **Why it matters**: <one sentence>\n"
    "- **Fix**:\n"
    "    <short corrected snippet>\n\n"

    "CRITICAL findings start with '⚠️ URGENT: '. Hardcoded secrets must "
    "instruct the user to move them to env vars/secrets IMMEDIATELY. "
    "Never say 'no issues' unless you scanned every line."
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