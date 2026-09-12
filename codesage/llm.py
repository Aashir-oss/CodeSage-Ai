"""Groq LLM wrapper - CodeSage tutor with domain guard + improvement mode."""
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
# SYSTEM PROMPTS  (built as concatenated strings to avoid quoting issues)
# ================================================================
BASE_SYSTEM = (
    "You are CodeSage, an expert Python code tutor embedded inside a "
    "code-review app.\n\n"
    "YOUR ONLY JOB is to help the user understand, review, improve, debug, "
    "or extend the Python project they uploaded. You are grounded in THEIR "
    "code - never give generic advice.\n\n"
    "DOMAIN RESTRICTION (very important):\n"
    "If the user's question is NOT about their uploaded code, Python, "
    "programming, software design, debugging, or code improvement, reply "
    "EXACTLY with this sentence and nothing else: "
    "\"I'm restricted to helping with your uploaded project. Please ask me "
    "something about your code - explaining a function, finding "
    "improvements, fixing issues, or adding new features.\"\n"
    "Do NOT answer general knowledge, news, trivia, math, personal "
    "questions, or anything unrelated to their code.\n\n"
    "STYLE RULES:\n"
    "- Plain English. Be concise but specific.\n"
    "- ALWAYS cite file names and line numbers when referring to code.\n"
    "- When suggesting code, show a short snippet in a fenced code block.\n"
    "- Match the project's existing style for any code you write.\n"
    "- Never invent file names or functions not present in the context.\n\n"
    "Project style: {style}"
)

IMPROVEMENT_ADDENDUM = (
    "\n\nIMPROVEMENT / SUGGESTION MODE.\n"
    "The user is asking for enhancements, tips, or improvements. "
    "Provide 3 to 6 concrete improvements. For each, use this exact format:\n\n"
    "### Short title\n"
    "- Where: <file> - line <number>\n"
    "- Why: <one-sentence rationale>\n"
    "- How:\n"
    "    <suggested code, matching project style>\n\n"
    "Focus on real things you can see: missing docstrings, bare excepts, "
    "long functions, duplicate logic, unclear names, missing type hints, "
    "potential crashes, security issues, dead code, better library choices. "
    "Never say 'I need more information' - use whatever code was retrieved."
)

THINKING_ADDENDUM = (
    "\n\nTHINKING MODE IS ON. Before answering:\n"
    "1. Analyze the overall architecture implied by the retrieved code.\n"
    "2. Identify dependencies, patterns, and conventions.\n"
    "3. Consider edge cases and best practices.\n"
    "4. Give a structured, thorough answer with sections.\n"
    "Use markdown headers (###) and bullets."
)

FEATURE_ADDENDUM = (
    "\n\nFEATURE-ADDITION MODE.\n"
    "The user wants to ADD or IMPLEMENT something new. Your job:\n"
    "1. Identify the best place to add it (specific file + line).\n"
    "2. Write the exact code (in the project's style).\n"
    "3. Explain how to integrate it (imports, calls, config).\n"
    "4. Show a short example.\n"
    "Format with sections: ### Where to add - ### Code - ### Integration"
)


# ================================================================
# INTENT DETECTION
# ================================================================
ADD_KEYWORDS = (
    "add", "implement", "create", "insert", "introduce",
    "how do i add", "how to add", "how can i add", "i want to add",
    "integrate", "extend",
)

IMPROVE_KEYWORDS = (
    "improve", "improvement", "enhance", "enhancement", "refactor",
    "optimi", "suggestion", "suggest", "tips", "clean up", "cleanup",
    "review", "what can be done", "what can i do", "make it better",
)


def detect_intent(question: str) -> str:
    q = question.lower().strip()
    if any(k in q for k in IMPROVE_KEYWORDS):
        return "improve"
    if any(k in q for k in ADD_KEYWORDS):
        return "add"
    if q.startswith(("what", "why", "explain", "how does", "how is", "describe")):
        return "explain"
    if "fix" in q or "bug" in q or "error" in q:
        return "fix"
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
        "Explain this " + func["type"] + " to a student.\n\n"
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
        "Student question: " + question + "\n\n"
        "Detected intent: " + intent + "\n\n"
        "Relevant context retrieved from their project:\n" + context
        + history_block + "\n\n"
        "Answer using ONLY this project's code and docs as evidence. "
        "Cite file + line (or page). If the question is not about this "
        "project or programming, apply the DOMAIN RESTRICTION."
    )

    return _chat(
        system, user_prompt,
        temperature=0.3 if thinking else 0.2,
        max_tokens=2500 if thinking else 1600,
    )


def deep_analysis(question: str, hits: list, style: str, all_items: list) -> str:
    system = BASE_SYSTEM.format(style=style) + THINKING_ADDENDUM
    if detect_intent(question) == "improve":
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
        "Retrieved relevant code:\n" + context + "\n\n"
        "Provide a thorough analysis with sections:\n"
        "### Overview\n"
        "### Architecture & Patterns\n"
        "### Concrete Improvements (with file + line + code)\n"
        "### Risks\n"
        "### Recommendations\n"
        "Cite files and line numbers. If the request is not about this "
        "project or programming, apply the DOMAIN RESTRICTION."
    )
    return _chat(system, user, temperature=0.35, max_tokens=3000)


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