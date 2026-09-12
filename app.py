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

# ==============================================================
# SETUP
# ==============================================================
load_dotenv()

st.set_page_config(
    page_title="CodeSage", page_icon="🧠",
    layout="wide", initial_sidebar_state="expanded",
)

ui.inject_css()

# ---- Auth placeholder (teammate plugs in later) ----
user_id  = st.session_state.get("user_id", "guest")
username = st.session_state.get("username", "guest")

# ---- Session defaults ----
DEFAULTS = {
    "project": None, "chat_history": [],
    "explain_result": None, "issue_explanation": None,
    "_last_upload": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ==============================================================
# PIPELINE
# ==============================================================
def run_pipeline_from_zip(zip_file, user_id: str):
    """Upload a ZIP → extract → parse → embed → analyze → learn style."""
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    zip_path = upload_dir / zip_file.name
    zip_path.write_bytes(zip_file.getbuffer())

    with st.status("🚀 Analyzing your project...", expanded=True) as status:
        st.write("📦 Extracting ZIP...")
        try:
            root = file_manager.extract_zip(str(zip_path), str(upload_dir / "current"))
        except Exception as e:
            status.update(label=f"❌ Extraction failed: {e}", state="error")
            return

        _finish_pipeline(root, user_id, status)


def run_pipeline_from_files(uploaded_files, user_id: str):
    """Upload individual files → save → parse → embed → analyze → learn style."""
    upload_dir = Path("data/uploads") / "current"
    if upload_dir.exists():
        import shutil
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    with st.status("🚀 Analyzing your files...", expanded=True) as status:
        st.write(f"📥 Saving {len(uploaded_files)} file(s)...")
        for f in uploaded_files:
            file_manager.save_uploaded_file(f, str(upload_dir))
        _finish_pipeline(upload_dir, user_id, status)


def _finish_pipeline(root: Path, user_id: str, status):
    """Shared logic after files are on disk."""
    all_files = file_manager.find_all_supported_files(root)
    code_files, doc_files, img_files = all_files["code"], all_files["docs"], all_files["images"]

    if not code_files and not doc_files and not img_files:
        status.update(label="❌ No supported files found", state="error")
        return

    st.write(f"   ✓ {len(code_files)} code · {len(doc_files)} docs · {len(img_files)} images")

    # Parse code
    st.write("🧬 Parsing Python files...")
    code_items = parser.parse_project(code_files, root)
    st.write(f"   ✓ {len(code_items)} functions/classes")

    # Load docs
    st.write("📚 Loading documents...")
    doc_items = doc_loader.load_all_documents(doc_files, img_files)
    st.write(f"   ✓ {len(doc_items)} document chunks")

    items = code_items + doc_items

    # Embed
    st.write("🧠 Generating embeddings...")
    progress = st.progress(0, text="Loading model (first run may take a minute)...")
    try:
        indexed = rag.index_items(items, user_id)
    except Exception as e:
        status.update(label=f"❌ Embedding failed: {e}", state="error")
        return
    progress.progress(100, text="Indexed ✓")
    st.write(f"   ✓ Indexed **{indexed}** items")

    # Analyze
    st.write("🔧 Running Ruff + Bandit...")
    issues = analyzer.analyze_project(root) if code_files else []
    st.write(f"   ✓ {len(issues)} issues")

    # Style
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


# ==============================================================
# SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown("## 🧠 CodeSage")
    st.caption("AI-powered code tutor")

    upload_mode = st.radio(
        "Upload mode",
        ["ZIP project", "Individual files"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if upload_mode == "ZIP project":
        uploaded_zip = st.file_uploader(
            "📁 Upload Python project (.zip)",
            type=["zip"], key="uploader_zip",
        )
        if uploaded_zip and st.session_state.get("_last_upload") != uploaded_zip.name:
            st.session_state["_last_upload"] = uploaded_zip.name
            run_pipeline_from_zip(uploaded_zip, user_id)
    else:
        uploaded_files = st.file_uploader(
            "📁 Upload files",
            type=["py", "pdf", "docx", "txt", "md", "png", "jpg", "jpeg"],
            accept_multiple_files=True, key="uploader_files",
        )
        if uploaded_files:
            signature = "|".join(sorted(f.name for f in uploaded_files))
            if st.session_state.get("_last_upload") != signature:
                st.session_state["_last_upload"] = signature
                run_pipeline_from_files(uploaded_files, user_id)

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
                st.session_state.project = None
                st.rerun()
        with col_b:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.project = None
                st.session_state["_last_upload"] = None
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