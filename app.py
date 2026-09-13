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
    "project": None,
    "chat_history": [],
    "explain_result": None,
    "issue_explanation": None,
    "_last_upload": None,
    "_last_paste": None,
    "_session_id": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


SUPPORTED_EXTS = set(file_manager.CODE_EXTS) | \
                 set(file_manager.DOC_EXTS) | \
                 set(file_manager.IMAGE_EXTS) | {".zip"}


def _reset_session_for_new_code():
    st.session_state.project = None
    st.session_state.chat_history = []
    st.session_state.explain_result = None
    st.session_state.issue_explanation = None


# ==============================================================
# PIPELINE
# ==============================================================
def _run_pipeline(root: Path, user_id: str, status, session_id: str):
    all_files = file_manager.find_all_supported_files(root)
    code_files  = all_files["code"]
    doc_files   = all_files["docs"]
    img_files   = all_files["images"]
    unsupported = all_files.get("unsupported", [])

    if unsupported:
        names = [p.name for p in unsupported[:5]]
        st.warning(
            f"⚠️ Skipped {len(unsupported)} unsupported file(s): "
            + ", ".join(f"`{n}`" for n in names)
            + (f" and {len(unsupported) - 5} more" if len(unsupported) > 5 else "")
        )

    if not code_files and not doc_files and not img_files:
        status.update(label="❌ No supported files found", state="error")
        return

    st.write(
        f"   ✓ {len(code_files)} code · {len(doc_files)} docs · "
        f"{len(img_files)} images"
    )

    # ---- Language breakdown ----
    langs = {}
    for f in code_files:
        lang = file_manager.language_for(f)
        langs[lang] = langs.get(lang, 0) + 1
    if langs:
        st.write("   🌐 Languages: " + " · ".join(
            f"{l}×{n}" for l, n in sorted(langs.items())
        ))

    st.write("🧬 Parsing code files...")
    code_items = parser.parse_project(code_files, root)
    st.write(f"   ✓ {len(code_items)} code chunks")

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

    st.write("🔧 Running static analysis...")
    issues = analyzer.analyze_project(root) if code_files else []
    st.write(f"   ✓ {len(issues)} issues")

    st.write("📚 Learning project style...")
    style = style_learner.learn_style(root, code_files, code_items)
    st.write(f"   ✓ {style['summary']}")

    status.update(label="✅ Ready! Ask away.", state="complete", expanded=False)

    summary = file_manager.scan_summary(root)
    st.session_state.project = {
        "root": str(root),
        "session_id": session_id,
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
    st.session_state.chat_history = []
    st.session_state.explain_result = None
    st.session_state.issue_explanation = None
    st.toast(f"Indexed {indexed} items ✓", icon="✅")


def run_pipeline_from_files(uploaded_files, user_id: str):
    _reset_session_for_new_code()

    upload_dir = Path("data/uploads") / "current"
    if upload_dir.exists():
        import shutil
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    session_id = "upload::" + "|".join(sorted(f.name for f in uploaded_files))

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
            _run_pipeline(root, user_id, status, session_id)
        else:
            _run_pipeline(upload_dir, user_id, status, session_id)


def run_pipeline_from_paste(code_text: str, user_id: str):
    _reset_session_for_new_code()

    paste_dir = Path("data/uploads") / "current"
    if paste_dir.exists():
        import shutil
        shutil.rmtree(paste_dir)
    paste_dir.mkdir(parents=True, exist_ok=True)

    # ---- Auto-detect the language ----
    lang = file_manager.detect_language_from_text(code_text)

    if lang == "unknown":
        st.error(
            "⚠️ **I cannot detect a supported language in the pasted text.**\n\n"
            "Please make sure your code is in one of these languages:\n\n"
            + " · ".join(f"`{x}`" for x in sorted(file_manager.SUPPORTED_LANGUAGES))
        )
        return

    ext = file_manager.extension_for_language(lang)
    code_file = paste_dir / f"pasted_code{ext}"
    code_file.write_text(code_text, encoding="utf-8")

    session_id = "paste::" + str(hash(code_text))

    with st.status("🚀 Analyzing pasted code...", expanded=True) as status:
        st.write(f"💾 Detected **{lang}** — saved as `pasted_code{ext}`")
        _run_pipeline(paste_dir, user_id, status, session_id)


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
            type=[
                "zip", "py",
                "js", "jsx", "ts", "tsx", "mjs", "cjs",
                "java", "kt", "kts", "scala",
                "cs",
                "c", "cpp", "cc", "cxx", "h", "hpp", "hxx",
                "go", "rs", "rb", "php", "swift",
                "html", "htm", "xml", "css", "scss", "sass", "less",
                "vue", "svelte",
                "sql", "sh", "bash", "ps1",
                "pdf", "docx", "txt", "md", "rst",
                "png", "jpg", "jpeg", "bmp", "gif",
            ],
            accept_multiple_files=True,
            key="uploader",
            help=(
                "Upload a ZIP or individual files. Supported: Python, "
                "JavaScript, TypeScript, Java, C#, C, C++, Go, Rust, Ruby, "
                "PHP, Swift, Kotlin, Scala, SQL, Shell, PowerShell, HTML, XML, "
                "CSS, SCSS, Vue, Svelte — plus PDF, DOCX, TXT, MD, and images."
            ),
        )

        if uploaded_files:
            bad = [f.name for f in uploaded_files
                   if Path(f.name).suffix.lower() not in SUPPORTED_EXTS]
            if bad:
                st.error(
                    "⚠️ **Unsupported file format(s):**\n\n"
                    + "\n".join(f"- `{n}`" for n in bad)
                    + "\n\n**Supported code:** `.py`, `.js`, `.ts`, `.jsx`, "
                    "`.tsx`, `.java`, `.cs`, `.c`, `.cpp`, `.h`, `.go`, "
                    "`.rs`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`, "
                    "`.html`, `.xml`, `.css`, `.scss`, `.vue`, `.svelte`, "
                    "`.sql`, `.sh`, `.ps1`\n\n"
                    "**Docs:** `.pdf`, `.docx`, `.txt`, `.md`\n\n"
                    "**Images:** `.png`, `.jpg`, `.jpeg`"
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
            "✍️ Paste your code",
            height=280,
            placeholder=(
                "# Paste code in any supported language:\n"
                "# Python, JavaScript, TypeScript, Java, C#, C, C++, Go,\n"
                "# Rust, Ruby, PHP, Swift, Kotlin, SQL, HTML, CSS, and more\n"
            ),
            key="paste_area",
        )

        if not pasted.strip() and st.session_state.get("_last_paste"):
            st.session_state["_last_paste"] = None
            _reset_session_for_new_code()

        if st.button("🚀 Analyze", use_container_width=True, type="primary"):
            if pasted and pasted.strip():
                new_sig = str(hash(pasted))
                if st.session_state.get("_last_paste") != new_sig:
                    st.session_state["_last_paste"] = new_sig
                    st.session_state["_last_upload"] = None
                    run_pipeline_from_paste(pasted, user_id)
                else:
                    st.info("This exact code is already indexed. Edit it to re-analyze.")
            else:
                st.warning("Paste some code first.")

    # ---------- Project stats ----------
    project = st.session_state.project

    if project:
        st.markdown("---")
        st.markdown("##### 📊 Project Stats")
        ui.render_metrics(project)

        st.markdown("---")
        st.markdown("##### ⚙️ Actions")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.explain_result = None
            st.session_state.issue_explanation = None
            st.rerun()

    st.markdown("---")
    st.markdown(f"👤 **{username}**")
    if st.button("⏻ Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ==============================================================
# MAIN — Only Chat
# ==============================================================
project = st.session_state.project

if not project:
    ui.welcome_screen()
else:
    ui.tab_chat(user_id, project)