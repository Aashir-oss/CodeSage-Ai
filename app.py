"""
CodeSage — AI code tutor for students.
Run with:  streamlit run app.py
"""
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from codesage import (
    file_manager,
    parser,
    doc_loader,
    rag,
    analyzer,
    style_learner,
    ui,
)

load_dotenv()

st.set_page_config(
    page_title="CodeSage", page_icon="🧠",
    layout="wide", initial_sidebar_state="expanded",
)

ui.inject_css()

user_id  = st.session_state.get("user_id", "guest")
username = st.session_state.get("username", "guest")

DEFAULTS = {
    "project": None, "chat_history": [],
    "explain_result": None, "issue_explanation": None,
    "_last_upload": None, "_last_paste": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


SUPPORTED_EXTS = {
    ".zip",
    ".py",
    ".pdf", ".docx", ".txt", ".md", ".rst",
    ".png", ".jpg", ".jpeg", ".bmp", ".gif",
}


# ==============================================================
# PIPELINE
# ==============================================================
def _run_pipeline(root: Path, user_id: str, status):
    all_files = file_manager.find_all_supported_files(root)
    code_files = all_files["code"]
    doc_files  = all_files["docs"]
    img_files  = all_files["images"]

    if not code_files and not doc_files and not img_files:
        status.update(label="❌ No supported files found", state="error")
        return

    st.write(f"   ✓ {len(code_files)} code · {len(doc_files)} docs · {len(img_files)} images")

    st.write("🧬 Parsing Python files...")
    code_items = parser.parse_project(code_files, root)
    st.write(f"   ✓ {len(code_items)} functions/classes")

    st.write("📚 Loading documents...")
    doc_items = doc_loader.load_all_documents(doc_files, img_files)
    st.write(f"   ✓ {len(doc_items)} document chunks")

    items = code_items + doc_items

    st.write("🧠 Generating embeddings...")
    progress = st.progress(0, text="Loading model (first run may take a minute)...")
    try:
        indexed = rag.index_items(items, user_id)
    except Exception as e:
        status.update(label=f"❌ Embedding failed: {e}", state="error")
        return
    progress.progress(100, text="Indexed ✓")
    st.write(f"   ✓ Indexed **{indexed}** items")

    st.write("🔧 Running Ruff + Bandit...")
    issues = analyzer.analyze_project(root) if code_files else []
    st.write(f"   ✓ {len(issues)} issues")

    st.write("📚 Learning project style...")
    style = style_learner.learn_style(root, code_files, code_items)
    st.write(f"   ✓ {style['summary']}")

    status.update(label="✅ Ready! Ask away.", state="complete", expanded=False)

    summary = file_manager.scan_summary(root)
    st.session_state.project = {
        "root": str(root),
        "file_count": summary["file_count"],
        "doc_count": summary["doc_count"],
        "image_count": summary["image_count"],
        "total_loc": summary["total_loc"],
        "items": items,
        "func_count": sum(1 for i in code_items if i["type"] == "function"),
        "class_count": sum(1 for i in code_items if i["type"] == "class"),
        "issues": issues,
        "style": style,
        "indexed_count": indexed,
    }
    st.session_state.explain_result = None
    st.session_state.issue_explanation = None
    st.session_state.chat_history = []
    st.toast(f"Indexed {indexed} items ✓", icon="✅")


def run_pipeline_from_files(uploaded_files, user_id: str):
    upload_dir = Path("data/uploads") / "current"
    if upload_dir.exists():
        import shutil
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    with st.status("🚀 Analyzing your upload...", expanded=True) as status:
        st.write(f"📥 Saving {len(uploaded_files)} file(s)...")
        for f in uploaded_files:
            file_manager.save_uploaded_file(f, str(upload_dir))

        if len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith(".zip"):
            zip_path = upload_dir / uploaded_files[0].name
            st.write("📦 Extracting ZIP...")
            try:
                root = file_manager.extract_zip(str(zip_path), str(upload_dir / "extracted"))
            except Exception as e:
                status.update(label=f"❌ Extraction failed: {e}", state="error")
                return
            _run_pipeline(root, user_id, status)
        else:
            _run_pipeline(upload_dir, user_id, status)


def run_pipeline_from_paste(code_text: str, user_id: str):
    """Save pasted code as a file, then run the standard pipeline."""
    paste_dir = Path("data/uploads") / "current"
    if paste_dir.exists():
        import shutil
        shutil.rmtree(paste_dir)
    paste_dir.mkdir(parents=True, exist_ok=True)

    code_file = paste_dir / "pasted_code.py"
    code_file.write_text(code_text, encoding="utf-8")

    with st.status("🚀 Analyzing pasted code...", expanded=True) as status:
        st.write("💾 Saved pasted code as pasted_code.py")
        _run_pipeline(paste_dir, user_id, status)


# ==============================================================
# SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown("## 🧠 CodeSage")
    st.caption("AI-powered code tutor")

    input_mode = st.radio(
        "How do you want to give your code?",
        ["📁 Upload files", "✍️ Paste code"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ---------- MODE 1: Upload ----------
    if input_mode == "📁 Upload files":
        uploaded_files = st.file_uploader(
            "📁 Upload project or files",
            type=["zip", "py", "pdf", "docx", "txt", "md",
                  "png", "jpg", "jpeg", "bmp", "gif"],
            accept_multiple_files=True,
            key="uploader",
            help="ZIP the whole project, or upload individual files.",
        )

        if uploaded_files:
            bad = [f.name for f in uploaded_files
                   if Path(f.name).suffix.lower() not in SUPPORTED_EXTS]
            if bad:
                st.error(
                    "⚠️ **Unsupported file format(s):**\n\n"
                    + "\n".join(f"- `{n}`" for n in bad)
                    + "\n\n**Allowed formats:**\n\n"
                    "`.zip` · `.py` · `.pdf` · `.docx` · `.txt` · `.md` · "
                    "`.png` · `.jpg` · `.jpeg`"
                )
            else:
                signature = "|".join(sorted(f.name for f in uploaded_files))
                if st.session_state.get("_last_upload") != signature:
                    st.session_state["_last_upload"] = signature
                    st.session_state["_last_paste"] = None
                    run_pipeline_from_files(uploaded_files, user_id)

    # ---------- MODE 2: Paste code ----------
    else:
        pasted = st.text_area(
            "✍️ Paste your Python code",
            height=280,
            placeholder=(
                "# Paste your Python code here\n"
                "def hello():\n"
                "    print('Hi')\n"
            ),
            key="paste_area",
        )

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("🚀 Analyze", use_container_width=True, type="primary"):
                if pasted and pasted.strip():
                    st.session_state["_last_paste"] = hash(pasted)
                    st.session_state["_last_upload"] = None
                    run_pipeline_from_paste(pasted, user_id)
                else:
                    st.warning("Paste some code first.")

    # ---------- Project stats ----------
    project = st.session_state.project

    if project:
        st.markdown("---")
        st.markdown("##### 📊 Project Stats")
        ui.render_metrics(project)

        st.markdown("---")
        ui.render_style(project["style"])

        st.markdown("---")
        st.markdown("##### ⚙️ Actions")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("♻️ Re-index", use_container_width=True):
                st.session_state["_last_upload"] = None
                st.session_state["_last_paste"] = None
                st.session_state.project = None
                st.rerun()
        with col_b:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.project = None
                st.session_state["_last_upload"] = None
                st.session_state["_last_paste"] = None
                st.session_state.chat_history = []
                st.rerun()

        ui.download_report(project)

    st.markdown("---")
    st.markdown(f"👤 **{username}**")
    if st.button("⏻ Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ==============================================================
# MAIN
# ==============================================================
project = st.session_state.project

if not project:
    ui.welcome_screen()
else:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Chat", "🔍 Search", "💡 Explain", "🔧 Issues"]
    )
    with tab1:
        ui.tab_chat(user_id, project)
    with tab2:
        ui.tab_search(user_id, project)
    with tab3:
        ui.tab_explain(user_id, project)
    with tab4:
        ui.tab_issues(user_id, project)