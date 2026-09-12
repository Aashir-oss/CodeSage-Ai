"""Groq LLM wrapper with thinking mode + feature-addition support."""
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
            raise RuntimeError("GROQ_API_KEY missing from .env")
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


BASE_SYSTEM = (
    "You are CodeSage, an expert Python code tutor for students. "
    "Explain clearly in plain English. Cite file names and line numbers "
    "whenever you refer to code. Respect the project's existing style.\n\n"
    "Project style: {style}"
)

THINKING_ADDENDUM = (
    "\n\nTHINKING MODE IS ON. Before answering:\n"
    "1. Analyze the overall architecture implied by the retrieved code.\n"
    "2. Identify dependencies, patterns, and conventions.\n"
    "3. Consider edge cases and best practices.\n"
    "4. Then give a structured, thorough answer with sections.\n"
    "Use markdown headers (###) and bullets. Include code snippets with "
    "file + line references."
)

FEATURE_ADDENDUM = (
    "\n\nFEATURE-ADDITION MODE. The user wants to ADD or IMPLEMENT something "
    "new in their codebase. Your job:\n"
    "1. Identify the best place to add it (specific file + line).\n"
    "2. Write the exact code to add (in the project's style).\n"
    "3. Explain how to integrate it (imports, calls, config).\n"
    "4. Show a short example of it working.\n"
    "Format with sections: ### Where to add · ### Code · ### Integration"
)

ADD_KEYWORDS = (
    "add", "implement", "create", "insert", "introduce",
    "how do i add", "how to add", "how can i add", "i want to add",
    "integrate", "extend",
)


def detect_intent(question: str) -> str:
    q = question.lower().strip()
    if any(k in q for k in ADD_KEYWORDS):
        return "add"
    if q.startswith(("what", "why", "explain", "how does", "how is", "describe")):
        return "explain"
    if "fix" in q or "bug" in q or "error" in q:
        return "fix"
    return "search"


def explain_function(func: dict, style: str) -> str:
    sys_p = BASE_SYSTEM.format(style=style)
    user = (
        f"Explain this {func['type']} to a student.\n\n"
        f"File: {func['file']} (lines {func['line_start']}-{func['line_end']})\n"
        f"Name: {func['name']}\n\n"
        f"```python\n{func['text']}\n```\n\n"
        f"Cover: what it does, its inputs, its outputs, and any obvious "
        f"issues (crashes, edge cases)."
    )
    return _chat(sys_p, user)


def answer_question(question: str, hits: list[dict], style: str,
                    thinking: bool = False,
                    history: list[dict] | None = None) -> str:
    intent = detect_intent(question)
    system = BASE_SYSTEM.format(style=style)
    if thinking:
        system += THINKING_ADDENDUM
    if intent == "add":
        system += FEATURE_ADDENDUM

    ctx_blocks = []
    for h in hits:
        if h["kind"] == "code":
            ctx_blocks.append(
                f"--- [CODE] {h['file']} :: {h['name']} "
                f"(lines {h['line_start']}-{h['line_end']}) ---\n{h['text']}"
            )
        else:
            ctx_blocks.append(
                f"--- [DOC] {h['file']} page {h.get('page', 1)} ---\n{h['text']}"
            )
    context = "\n\n".join(ctx_blocks) or "(no matching context found)"

    history_block = ""
    if history:
        recent = history[-6:]
        history_block = "\n\nPrevious conversation:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}" for m in recent
        )

    user_prompt = (
        f"Student question: {question}\n\n"
        f"Intent: {intent}\n\n"
        f"Relevant context retrieved from their project:\n{context}"
        f"{history_block}\n\n"
        f"Answer using ONLY this project's code and docs as evidence. "
        f"Cite file + line (or page). If context is insufficient, say so."
    )

    return _chat(
        system, user_prompt,
        temperature=0.3 if thinking else 0.2,
        max_tokens=2500 if thinking else 1200,
    )


def deep_analysis(question: str, hits: list[dict], style: str,
                  all_items: list[dict]) -> str:
    system = BASE_SYSTEM.format(style=style) + THINKING_ADDENDUM

    file_tree = sorted(set(i["file"] for i in all_items))
    summary = (
        f"Files in project ({len(file_tree)}):\n"
        + "\n".join(f"- {f}" for f in file_tree[:50])
    )

    ctx_blocks = []
    for h in hits:
        ctx_blocks.append(f"--- {h['file']} :: {h['name']} ---\n{h['text']}")
    context = "\n\n".join(ctx_blocks)

    user = (
        f"Deep analysis request: {question}\n\n"
        f"Project structure:\n{summary}\n\n"
        f"Retrieved relevant code:\n{context}\n\n"
        f"Provide a thorough architectural analysis:\n"
        f"### Overview\n"
        f"### Architecture & Patterns\n"
        f"### Dependencies & Data Flow\n"
        f"### Issues & Risks\n"
        f"### Recommendations\n"
        f"### Code Examples\n"
        f"Be specific — cite files and line numbers."
    )
    return _chat(system, user, temperature=0.35, max_tokens=3000)


def explain_issue(issue: dict, style: str) -> str:
    sys_p = BASE_SYSTEM.format(style=style)
    user = (
        f"Explain this static-analysis finding and give a one-line fix.\n\n"
        f"Tool: {issue['tool']}\nFile: {issue['file']} line {issue['line']}\n"
        f"Code: {issue['code']}\nMessage: {issue['message']}\n\n"
        f"Format exactly:\nProblem: <one sentence>\nFix: <one-line suggestion>"
    )
    return _chat(sys_p, user, temperature=0.1)