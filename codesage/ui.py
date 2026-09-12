"""Reusable UI components for CodeSage."""
import json
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime


# ==============================================================
# CSS / HERO
# ==============================================================
def inject_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def hero():
    st.markdown(
        """
        <div class="cs-hero">
            <h1>🧠 CodeSage</h1>
            <p>Upload your Python project + docs. Ask anything. Get AI answers
            grounded in <em>your</em> code, with file + line citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================
# SIDEBAR
# ==============================================================
def render_metrics(project: dict):
    stats = [
        ("📄 Code files", project["file_count"]),
        ("📚 Docs",       project["doc_count"]),
        ("🖼️ Images",     project["image_count"]),
        ("🔧 Functions",  project["func_count"]),
        ("🏛️ Classes",    project["class_count"]),
        ("📏 LOC",        project["total_loc"]),
        ("⚠️ Issues",     len(project["issues"])),
        ("🧬 Indexed",    project["indexed_count"]),
    ]
    for label, value in stats:
        st.markdown(
            f'<div class="cs-metric"><span>{label}</span><span>{value}</span></div>',
            unsafe_allow_html=True,
        )


def render_style(style: dict):
    st.markdown("##### 🎨 Detected Style")
    extra = ""
    if style.get("common_raises"):
        extra = f'<div class="cs-card-meta">Raises: {", ".join(style["common_raises"])}</div>'
    st.markdown(
        f'<div class="cs-card">'
        f'<div class="cs-card-title">Naming: '
        f'<span class="cs-badge cs-badge-style">{style["naming"]}</span></div>'
        f'<div class="cs-card-meta">Errors: {style["error_pattern"]}</div>'
        f'<div class="cs-card-meta">Docs: {int(style["doc_ratio"] * 100)}% documented</div>'
        f"{extra}</div>",
        unsafe_allow_html=True,
    )


def download_report(project: dict):
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "stats": {
            "files": project["file_count"], "docs": project["doc_count"],
            "images": project["image_count"], "functions": project["func_count"],
            "classes": project["class_count"], "loc": project["total_loc"],
            "indexed": project["indexed_count"],
        },
        "style": project["style"], "issues": project["issues"],
    }
    st.download_button(
        "📥 Download Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"codesage_report_{datetime.now():%Y%m%d_%H%M%S}.json",
        mime="application/json",
        use_container_width=True,
    )


# ==============================================================
# CARDS
# ==============================================================
def code_card(name, file, line_start, line_end, source,
              item_type="function", score=None):
    icon = "🔧" if item_type == "function" else "🏛️"
    score_html = (
        f'<span class="cs-card-meta">score {score:.3f}</span>'
        if score is not None else ""
    )
    st.markdown(
        f"""
        <div class="cs-card">
            <div class="cs-card-header">
                <span>{icon} {name}</span>{score_html}
            </div>
            <div class="cs-card-meta">{file} · lines {line_start}–{line_end}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(source, language="python")


def doc_card(name, file, page, text, score=None):
    score_html = (
        f'<span class="cs-card-meta">score {score:.3f}</span>'
        if score is not None else ""
    )
    preview = text[:400] + ("..." if len(text) > 400 else "")
    st.markdown(
        f"""
        <div class="cs-card">
            <div class="cs-card-header">
                <span>📄 {name}</span>{score_html}
            </div>
            <div class="cs-card-meta">{file} · page {page}</div>
            <div style="color:#cbd5e1;font-size:0.85rem;margin-top:8px;white-space:pre-wrap;">{preview}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def issue_card(issue: dict):
    sev = issue.get("severity", "low").lower()
    badge_cls = {"high": "cs-badge-high",
                 "medium": "cs-badge-medium",
                 "low": "cs-badge-low"}.get(sev, "cs-badge-low")
    st.markdown(
        f"""
        <div class="cs-card">
            <div class="cs-card-header">
                <span>{issue['tool'].upper()} · {issue['code']}</span>
                <span class="cs-badge {badge_cls}">{sev.upper()}</span>
            </div>
            <div class="cs-card-title">{issue['message']}</div>
            <div class="cs-card-meta">{issue['file']} · line {issue['line']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================
# WELCOME
# ==============================================================
def welcome_screen():
    hero()
    st.info(
        "👈 Upload a `.zip` **or** drag in individual files "
        "(`.py`, `.pdf`, `.docx`, `.txt`, `.md`, `.png`, `.jpg`) "
        "to get started."
    )


# ==============================================================
# TAB 1 — SEARCH
# ==============================================================
def tab_search(user_id: str, project: dict):
    from codesage import rag, llm

    st.markdown("### 🔍 Smart Code Search")
    st.caption("Ask about your codebase and get answers backed by your actual code.")

    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Ask a question",
            placeholder='e.g. "How is user input validated?"',
            key="search_query", label_visibility="collapsed",
        )
    with col2:
        go = st.button("Search", use_container_width=True, type="primary")

    st.caption("💡 Try: *input validation* · *file handling* · *error handling*")

    if go and query:
        with st.spinner("Searching your codebase..."):
            from codesage import llm as _llm
            intent = _llm.detect_intent(query)
            hits = _smart_retrieve(query, user_id, intent, project)
            if not hits:
                st.warning("No relevant code found. Try re-indexing or rephrasing.")
                return

            answer = llm.answer_question(query, hits, project["style"]["summary"])

        st.markdown("#### 💡 Answer")
        st.markdown(f'<div class="cs-card">{answer}</div>', unsafe_allow_html=True)

        st.markdown(f"#### 📎 Retrieved {len(hits)} source(s)")
        for h in hits:
            if h["kind"] == "code":
                code_card(h["name"], h["file"], h["line_start"], h["line_end"],
                          h["text"], h["type"], score=h.get("score"))
            else:
                doc_card(h["name"], h["file"], h.get("page", 1),
                         h["text"], score=h.get("score"))


# ==============================================================
# TAB 2 — EXPLAIN
# ==============================================================
def tab_explain(user_id: str, project: dict):
    from codesage import rag, llm

    st.markdown("### 💡 AI Code Explainer")
    st.caption("Pick any function or class and get a plain-English explanation.")

    all_items = rag.list_all(user_id)
    if not all_items:
        st.warning("No functions/classes indexed yet.")
        return

    options = {f"{i['type']} · {i['file']} :: {i['name']} (line {i['line_start']})": i
               for i in all_items}
    selected_key = st.selectbox("Choose an item", options=list(options.keys()),
                                key="explain_select", label_visibility="collapsed")
    selected = options[selected_key]

    if st.button("💡 Explain this", type="primary"):
        with st.spinner("Thinking..."):
            explanation = llm.explain_function(selected, project["style"]["summary"])
            st.session_state.explain_result = {"item": selected, "explanation": explanation}

    result = st.session_state.get("explain_result")
    if result and result["item"]["name"] == selected["name"]:
        st.markdown("#### 📖 Explanation")
        st.markdown(f'<div class="cs-card">{result["explanation"]}</div>',
                    unsafe_allow_html=True)
        st.markdown("#### 📝 Source")
        st.code(result["item"]["text"], language="python")


# ==============================================================
# TAB 3 — ISSUES
# ==============================================================
def tab_issues(user_id: str, project: dict):
    from codesage import llm
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

    st.markdown("### 🔧 Issue Detector + Fix Suggester")
    issues = project["issues"]

    if not issues:
        st.success("🎉 No issues found — nice work!")
        return

    st.caption(f"Found **{len(issues)}** issues. Click a row to explain + fix.")

    rows = [{
        "Tool": i["tool"], "File": i["file"], "Line": i["line"],
        "Code": i["code"], "Severity": i["severity"].upper(),
        "Message": i["message"],
    } for i in issues]
    df = pd.DataFrame(rows)

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection("single", use_checkbox=False)
    gb.configure_default_column(filter=True, sortable=True, resizable=True)
    gb.configure_column("Message", width=400)
    gb.configure_column("Line", width=80)
    gb.configure_column("Severity", width=110)

    grid = AgGrid(df, gridOptions=gb.build(),
                  update_mode=GridUpdateMode.SELECTION_CHANGED,
                  fit_columns_on_grid_load=False, height=350,
                  theme="streamlit", key="issues_grid")

    selected = grid.get("selected_rows")
    if selected is not None and len(selected) > 0:
        row = selected.iloc[0].to_dict() if hasattr(selected, "iloc") else selected[0]
        issue = next((i for i in issues
                      if i["file"] == row["File"] and i["line"] == row["Line"]
                      and i["message"] == row["Message"]), None)
        if issue:
            st.markdown("#### 🔎 Selected Issue")
            issue_card(issue)
            if st.button("🤖 Explain + Suggest Fix", type="primary"):
                with st.spinner("Analyzing..."):
                    st.session_state.issue_explanation = llm.explain_issue(
                        issue, project["style"]["summary"])
            if st.session_state.get("issue_explanation"):
                st.markdown(f'<div class="cs-card">{st.session_state.issue_explanation}</div>',
                            unsafe_allow_html=True)


# ==============================================================
# TAB 4 — CHAT (main experience, with thinking mode)
# ==============================================================
def render_sources(hits: list):
    if not hits:
        return
    with st.expander(f"📎 Sources ({len(hits)} retrieved)", expanded=False):
        for h in hits:
            if h["kind"] == "code":
                code_card(h["name"], h["file"], h["line_start"], h["line_end"],
                          h["text"], h["type"], score=h.get("score"))
            else:
                doc_card(h["name"], h["file"], h.get("page", 1),
                         h["text"], score=h.get("score"))


def _smart_retrieve(prompt: str, user_id: str, intent: str, project: dict = None) -> list:
    """
    Retrieve relevant context. Falls back to ALL indexed items when:
      - project is small (<=15 items), OR
      - retrieval returns fewer than 3 hits.

    This guarantees the LLM always sees code from small pasted snippets.
    """
    from codesage import rag

    # ---- Baseline retrieval ----
    if intent in ("improve", "error", "fix"):
        hits = rag.retrieve(prompt, user_id, top_k=8)
        hits += rag.retrieve(
            "functions with exceptions error handling security risks "
            "hardcoded values weak hashing injection division by zero "
            "unused imports bare except missing docstrings",
            user_id, top_k=8,
        )
        hits += rag.retrieve(
            "main entry point class definition long function helper "
            "utility validate read file command run",
            user_id, top_k=6,
        )
    else:
        hits = rag.retrieve(prompt, user_id, top_k=6)

    # ---- De-duplicate ----
    seen, unique = set(), []
    for h in hits:
        key = (h["file"], h["name"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(h)

    # ---- FALLBACK: pull everything if the corpus is small or retrieval was weak ----
    all_items = rag.list_all(user_id)
    total_items = len(all_items)

    # If the project has <=15 code chunks OR we got <3 hits → dump everything
    if total_items > 0 and (total_items <= 15 or len(unique) < 3):
        # Convert list_all output into retrieval-like dicts
        for item in all_items:
            key = (item["file"], item["name"])
            if key in seen:
                continue
            seen.add(key)
            unique.append({
                "kind": "code",
                "type": item.get("type", "function"),
                "name": item.get("name", ""),
                "file": item.get("file", ""),
                "line_start": item.get("line_start", 0),
                "line_end": item.get("line_start", 0) + item.get("text", "").count("\n"),
                "text": item.get("text", ""),
                "docstring": "",
                "score": None,
            })

    # ---- Diversity: max 3 chunks per file, cap total at 12 ----
    files_seen, diversified = {}, []
    for h in unique:
        files_seen.setdefault(h["file"], 0)
        if files_seen[h["file"]] < 3:
            diversified.append(h)
            files_seen[h["file"]] += 1
        if len(diversified) >= 12:
            break

    return diversified if diversified else unique

def tab_chat(user_id: str, project: dict):
    """Chat with input pinned at the bottom, history scrolls above."""
    from codesage import llm

    st.markdown("### 💬 Ask CodeSage")
    st.caption("Chat with your project. Ask to explain, find, fix, or add features.")

    c1, c2 = st.columns([1, 2])
    with c1:
        thinking = st.toggle(
            "🧠 Deep Thinking Mode", value=False,
            help="Full architectural analysis. Slower but thorough.",
        )
    with c2:
        st.caption(
            "💡 *Try:* *what improvements can be done?* · "
            "*add logging to my main function* · *explain the auth flow*"
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---- Scrollable history container ----
    history_container = st.container()
    with history_container:
        if not st.session_state.chat_history:
            st.markdown(
                '<div style="text-align:center;color:#64748b;padding:40px 20px;">'
                '<div style="font-size:2rem;">💬</div>'
                '<p>Ask anything about your uploaded code to get started.</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    render_sources(msg["sources"])

    # ---- Input pinned at bottom (Streamlit auto-pins st.chat_input) ----
    prompt = st.chat_input("Ask anything about your code...")

    if prompt:
        # Append user message first so it renders in next rerun
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Compute answer
        with st.spinner("Thinking..." if not thinking else "Deep analysis in progress..."):
            intent = llm.detect_intent(prompt)
            hits = _smart_retrieve(prompt, user_id, intent, project)

            if thinking:
                answer = llm.deep_analysis(
                    prompt, hits, project["style"]["summary"], project["items"],
                )
            else:
                answer = llm.answer_question(
                    prompt, hits, project["style"]["summary"],
                    thinking=False,
                    history=st.session_state.chat_history[:-1],
                )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": hits,
        })

        # Rerun so the new messages render in the history container
        st.rerun()