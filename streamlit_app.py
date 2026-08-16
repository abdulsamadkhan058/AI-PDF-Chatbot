"""Professional Mltilingual AI PDF ChatBot."""
import base64
import hashlib
import html
import json
import os
import pickle
import re
import shutil
import sqlite3
import tempfile
import time
import edge_tts
from urllib.parse import urlparse
import streamlit as st
import whisper
from ddgs import DDGS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from answer_validator import validate_answer
from context_compressor import compress_context
from conversation_memory import ConversationMemory
from core.embeddings import load_embedding
from core.llm import load_llm
from language_utils import (
    detect_language,
    is_wrong_script,
    language_instruction,
    normalize_text,
)
from retrieval import build_source_bm25, retrieve_documents

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "AI PDF Bot"
FAISS_FOLDER = "faiss_db"
CHUNKS_FILE = "chunks.pkl"
PDF_INFO_FILE = "pdf_info.pkl"
UPLOAD_FOLDER = "uploads"
DB_FILE = "chat_history.db"

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
HERO_LOGO_PATH = os.path.join(ASSETS_DIR, "bot_logo.png")
BOT_AVATAR_PATH = os.path.join(ASSETS_DIR, "chat_bot_avatar.png")
USER_AVATAR_PATH = os.path.join(ASSETS_DIR, "user_avatar.png")

BOT_AVATAR = (
    BOT_AVATAR_PATH if os.path.exists(BOT_AVATAR_PATH)
    else HERO_LOGO_PATH if os.path.exists(HERO_LOGO_PATH)
    else "🤖"
)
USER_AVATAR = USER_AVATAR_PATH if os.path.exists(USER_AVATAR_PATH) else "🧑"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_K = 6
FINAL_K = 3

REFUSALS = {
    "English": "I don't know based on the provided PDF context.",
    "Urdu": "مجھے فراہم کردہ PDF کے سیاق و سباق کی بنیاد پر اس کا جواب معلوم نہیں۔",
    "Roman Urdu": "Mujhe diye gaye PDF context ki bunyaad par iska jawab maloom nahi.",
    "Arabic": "لا أعرف الإجابة بناءً على سياق ملف PDF المقدم.",
}

def refusal(language):
    return REFUSALS.get(language, REFUSALS["English"])

def is_refusal(text):
    value = normalize_text(text).lower()
    return any(value == normalize_text(x).lower() for x in REFUSALS.values())

def inject_css():
    st.markdown(r"""
    <style>
    :root {
        --bg:#030816; --panel:#071329; --panel2:#0b1d3b;
        --border:#163966; --text:#f8fbff; --muted:#aebcd5;
        --accent:#267dff; --violet:#8d35ff; --cyan:#13ceff;
        --native-header-h:3.15rem; --hero-h:6.7rem;
    }
    .stApp {
        background:radial-gradient(circle at 55% -10%,#172a71 0,#050b1e 29%,#020714 70%) !important;
        color:var(--text) !important;
    }
    header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
        background:#030816 !important; color:#fff !important;
    }
    .block-container {
        max-width:1510px !important;
        padding:calc(var(--hero-h) + 1.15rem) 1.4rem 7.2rem !important;
    }
    .stApp p,.stApp span,.stApp label,.stMarkdown,.stMarkdown p,
    .stMarkdown li,[data-testid="stCaptionContainer"] {
        color:var(--text) !important;
    }
    [data-testid="stCaptionContainer"] { color:var(--muted) !important; }
    h1,h2,h3,h4 { color:#fff !important; }

    .hero {
        min-height:84px; border:0; border-bottom:1px solid #1e3a66; border-radius:0;
        padding:.85rem 1.8rem; background:linear-gradient(90deg,#081226,#0a1730 45%,#081226);
        margin-bottom:1.15rem; display:flex; align-items:center; justify-content:space-between;
        overflow:visible; box-sizing:border-box;
        position:fixed !important; top:var(--native-header-h); left:0; right:0;
        width:100%; z-index:999999;
        box-shadow:0 4px 18px rgba(0,0,0,.35);
    }
    .hero-title { display:flex; align-items:center; font-size:clamp(1.6rem,2.6vw,2.15rem); font-weight:750; letter-spacing:-.03em; color:#fff !important;
        background:linear-gradient(105deg,#f6fbff 25%,#c3cdf2 55%,#8fa3ef 88%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .bot-mark { display:inline-flex; width:64px; height:64px; margin-right:.85rem;
        align-items:center; justify-content:center; border-radius:14px; overflow:hidden;
        background:#0a1a38;
        border:1px solid #2c5a9c; box-shadow:0 4px 14px rgba(0,0,0,.4);
        vertical-align:middle; flex:0 0 auto; }
    .bot-mark img { width:100%; height:100%; object-fit:cover; object-position:center 30%; }
    .hero-sub { color:#9fb0cc !important; margin:.15rem 0 0 4.95rem; line-height:1.5; font-size:.92rem; font-weight:500; }
    .hero-badges { display:flex; gap:.6rem; flex-wrap:wrap; justify-content:flex-end; }
    .hero-badge { border:1px solid #26436e; border-radius:8px; padding:.5rem .85rem; font-weight:600; font-size:.83rem;
        background:#0c1c3a; color:#7fc4ff !important; letter-spacing:.01em; }
    .hero-badge.purple { color:#c99bff !important; background:#150f30; border-color:#372060; }

    /* The hero bar above is fixed to the viewport (full width) so it always
       stays visible while the chat scrolls, regardless of which inner
       Streamlit container actually owns the page scroll. Push the sidebar's
       content down so it isn't hidden underneath the fixed bar. */
    section[data-testid="stSidebar"] {
        padding-top:var(--hero-h) !important;
        background:linear-gradient(180deg,#07162d,#030b1b) !important;
        border-right:1px solid #1a3f70 !important;
    }
    section[data-testid="stSidebar"] * { color:#edf3ff !important; }

    .stButton>button,.stDownloadButton>button {
        width:100% !important; min-height:44px !important;
        border-radius:12px !important; border:1px solid #163760 !important;
        background:linear-gradient(135deg,#0c1e3c,#0a152c) !important; color:#f7faff !important;
        font-weight:650 !important;
    }
    .stButton>button:hover,.stDownloadButton>button:hover {
        border-color:#4ea8ff !important; background:linear-gradient(135deg,#2824a7,#147cf7) !important;
    }

    [data-testid="stFileUploader"] { background:#071a34 !important; border:1px solid #1d72c9 !important;
        border-radius:18px !important; padding:.45rem !important; box-shadow:0 0 26px #087dff33 !important; }
    [data-testid="stFileUploaderDropzone"] {
        background:linear-gradient(135deg,#082650,#07172f) !important; border:1px dashed #43a8ff !important;
        border-radius:14px !important; min-height:130px !important;
    }
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] button {
        color:#ffffff !important; opacity:1 !important; visibility:visible !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background:linear-gradient(135deg,#6540ff,#147ff5) !important; border:1px solid #72c5ff !important;
        border-radius:10px !important; font-weight:800 !important; text-shadow:none !important;
    }
    [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] { background:transparent !important; color:#fff !important; }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * { color:#fff !important; fill:#fff !important; }

    div[data-testid="stChatMessage"] {
        background:linear-gradient(135deg,#0b1b38,#07142b) !important; border:1px solid #1e477a !important;
        border-radius:19px !important; padding:1rem 1.15rem !important;
        margin:.78rem 0 !important; box-shadow:0 9px 25px rgba(0,0,0,.26);
    }
    div[data-testid="stChatMessage"] p { color:#f4f7ff !important; line-height:1.7 !important; }
    [data-testid="stChatMessageAvatar"] { border:1px solid #b2eaff !important; border-radius:14px !important;
        background:radial-gradient(circle at 50% 35%,#1ee2ff 0 14%,#142b71 17% 48%,#3d167d 76%) !important;
        box-shadow:inset -3px -4px 7px #0008,inset 2px 2px 5px #fff5,0 0 12px #1b8fff88 !important; }
    [data-testid="stChatMessageAvatar"] img { filter:drop-shadow(0 2px 2px #000a) saturate(1.25) !important; }

    [data-testid="stChatInput"] {
        background:linear-gradient(135deg,#0d193a,#06152c) !important; border:1px solid #2b9cff !important;
        border-radius:19px !important; box-shadow:0 0 20px #087dff44 !important;
    }
    [data-testid="stChatInput"] textarea {
        color:#fff !important; background:#111e31 !important;
        caret-color:#fff !important; font-size:16px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color:#aebbd0 !important; opacity:1 !important;
    }
    /* Streamlit renders a separate fixed bottom container. These rules remove
       the white desktop/mobile bar around the otherwise dark chat input. */
    [data-testid="stBottom"], [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInputContainer"] {
        background:#030816 !important; background-color:#030816 !important;
        border-color:#030816 !important; box-shadow:none !important;
    }
    [data-testid="stChatInputContainer"] { padding: .7rem 1rem 1rem !important; }
    [data-testid="stChatInput"] > div, [data-testid="stChatInput"] form,
    [data-testid="stChatInput"] textarea { background-color:#0a1933 !important; }
    [data-testid="stChatInput"] button { background:#1b6ff0 !important; color:#fff !important; }

    .answer-box {
        color:#f5f8ff !important; font-size:1.05rem; line-height:1.8;
        white-space:pre-wrap; overflow-wrap:anywhere;
    }
    .answer-box.rtl {
        direction:rtl; text-align:right;
        font-family:"Noto Naskh Arabic","Noto Sans Arabic","Segoe UI",Arial,sans-serif;
    }
    .status-box {
        background:#12213a; border:1px solid #355277; border-radius:11px;
        padding:.65rem .8rem; color:#dbe7f8 !important;
        margin:.4rem 0 .7rem;
    }
    .status-box * { color:#dbe7f8 !important; }

    /* Native Streamlit spinner / status widgets (shown while processing PDFs,
       loading the embedding model, transcribing audio, etc.) default to a
       white background — restyle to match the dark theme. */
    [data-testid="stSpinner"] {
        background:transparent !important;
    }
    [data-testid="stSpinner"] > div {
        background:#12213a !important; border:1px solid #355277 !important;
        border-radius:11px !important; box-shadow:none !important;
    }
    [data-testid="stSpinner"] * { color:#dbe7f8 !important; fill:#dbe7f8 !important; }
    [data-testid="stStatusWidget"], [data-testid="stStatusWidget"] > div,
    [data-testid="stExpander"], [data-testid="stExpander"] > div,
    [data-testid="stExpander"] summary {
        background:#12213a !important; border-color:#355277 !important;
        color:#dbe7f8 !important;
    }
    [data-testid="stExpander"] * { color:#dbe7f8 !important; }

    .pdf-card {
        background:linear-gradient(135deg,#092850,#071a38); border:1px solid #194b81;
        border-radius:13px; padding:.75rem .8rem; margin:.45rem 0; box-shadow:inset 0 1px #ffffff12;
    }
    .pdf-name { color:#fff !important; font-weight:750; overflow-wrap:anywhere; }
    .pdf-meta { color:#aab8cc !important; font-size:.78rem; margin-top:.2rem; }
    .uploaded-files { display:flex; flex-wrap:wrap; gap:.55rem; margin:.55rem 0 .85rem; }
    .uploaded-file { background:#0b274d; color:#fff !important; border:1px solid #3ea8ff;
        border-radius:10px; padding:.48rem .7rem; font-weight:700; box-shadow:0 3px 10px #0005; }

    /* Sources now render INSIDE the same .answer-box as the answer text
       (single flowing block per turn) instead of a separate boxed element
       below it. Keep them visually light: no card background/border, just
       a soft divider so it reads as a continuation of the same answer. */
    .sources-inline {
        margin-top:.9rem; padding-top:.7rem;
        border-top:1px dashed #26436e;
    }
    .sources-title {
        color:#b8c6db !important; font-size:.78rem; font-weight:800;
        letter-spacing:.06em; text-transform:uppercase;
        margin-bottom:.45rem;
    }
        .source-line {
        display:block !important;
        padding:.35rem 0;
        color:#edf3ff !important;
        font-size:.92rem;
        min-width:0;
        text-align:left !important;
        overflow-wrap:anywhere;
        word-break:break-word;
    }

    .source-line b {
        display:block !important;
        color:#f4f7ff !important;
        font-weight:700 !important;
        text-align:left !important;
        overflow-wrap:anywhere;
        word-break:break-word;
    }

    .source-meta {
        display:block !important;
        margin-top:2px !important;
        color:#9aaac0 !important;
        font-size:.8rem;
        text-align:left !important;
        white-space:normal !important;
    }

    .source-line a {
        display:inline-block !important;
        color:#7fc4ff !important;
        text-decoration:none;
        font-weight:650;
        text-align:left !important;
    }

    .source-snippet {
        display:block !important;
        margin-top:3px;
        color:#9aaac0 !important;
        font-size:.82rem;
        line-height:1.5;
        text-align:left !important;
    }

    .pill {
        display:inline-block; color:#dce6f5 !important; background:#132239;
        border:1px solid #2c4565; border-radius:999px;
        padding:.3rem .6rem; margin:.15rem .1rem; font-size:.78rem;
    }
    .empty-state {
        border:1px solid #29405f; background:#101b2d;
        border-radius:16px; padding:1.2rem; color:#dbe6f5 !important;
    }
    .empty-state * { color:#dbe6f5 !important; }
    .pdf-rack { display:flex; gap:1rem; overflow-x:auto; padding:1rem; margin:.35rem 0 1.2rem;
        border:2px solid #0b9cff; border-radius:22px; background:linear-gradient(135deg,#061a36,#081329); box-shadow:0 0 25px #087dff40; }
    .pdf-tile { min-width:172px; padding:1rem .7rem .65rem; border-radius:15px; text-align:center; border:1px solid #278ee9;
        background:radial-gradient(circle at 50% 12%,#276dc2,#071b3b 63%); box-shadow:inset 0 1px #d9f2ff33,0 7px 15px #0007; }
    .pdf-tile:nth-child(2n) { background:radial-gradient(circle at 50% 12%,#1e9a6a,#062d30 63%); border-color:#28c78f; }
    .pdf-tile:nth-child(3n) { background:radial-gradient(circle at 50% 12%,#b67529,#311b08 63%); border-color:#dd9b3b; }
    .pdf-icon { font-size:2.75rem; filter:drop-shadow(0 7px 4px #0009); margin-bottom:.4rem; }
    .pdf-tile-name { font-weight:800; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:white !important; }
    .pdf-tile-meta { color:#d5e4ff !important; font-size:.83rem; margin-top:.25rem; }
    .chat-heading { font-size:1.6rem; font-weight:850; margin:1.1rem 0 .4rem; color:#fff !important; }
    .features { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.7rem; margin-top:1.8rem; }
    .feature { padding:.8rem; border:1px solid #173e6b; border-radius:13px; background:#07162d; color:#d8e8ff !important; }
    .feature b { display:block; color:white !important; margin-bottom:.25rem; }

    /* Mobile browsers resize their own UI (address bar / bottom nav bar)
       as the page scrolls, which changes the *actual* visible viewport
       height without firing a resize the app can react to. Streamlit's
       chat input is pinned using the viewport height, so on some mobile
       browsers/Streamlit versions it (and its microphone icon) end up
       positioned behind that browser UI and become invisible/unreachable
       (see streamlit/streamlit#11891, #11722, #14152). Forcing dvh
       ("dynamic viewport height") here, plus the safe-area inset, keeps
       the fixed bottom chat bar and hero bar aligned to the real visible
       area instead of the stale 100vh value.
    */
    @supports (height: 100dvh) {
        [data-testid="stAppViewContainer"], [data-testid="stMain"],
        [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
            min-height: auto;
        }
        .stApp { min-height: 100dvh !important; }
    }
    [data-testid="stBottomBlockContainer"], [data-testid="stChatInputContainer"] {
        padding-bottom: max(1rem, env(safe-area-inset-bottom)) !important;
    }

    @media (max-width:768px) {
        :root { --hero-h:5.7rem; }
        .block-container { padding:calc(var(--hero-h) + 1.1rem) .7rem 6.8rem !important; }
        .hero { display:block; min-height:auto; padding:.8rem .82rem; border-radius:0; }
        .hero-title { font-size:clamp(1.55rem,8.2vw,2rem); letter-spacing:-.04em; white-space:nowrap; }
        .bot-mark { width:49px; height:49px; font-size:1.65rem; border-radius:13px; }
        .hero-sub { margin:.3rem 0 0 3.55rem; font-size:.84rem; }
        .hero-badges { display:none; }
        div[data-testid="stChatMessage"] {
            border-radius:13px !important; padding:.7rem .72rem !important;
        }
        div[data-testid="stChatMessage"] p {
            font-size:.93rem !important; line-height:1.6 !important;
        }
        [data-testid="stFileUploaderDropzone"] { min-height:105px !important; }
        section[data-testid="stSidebar"] {
            width:285px !important; min-width:285px !important;
        }
        .answer-box { font-size:.95rem; }
        .pdf-rack { gap:.65rem; padding:.65rem; border-radius:16px; }
        .pdf-tile { min-width:133px; padding:.65rem .5rem; }
        .pdf-icon { font-size:2.15rem; }
        .features { grid-template-columns:repeat(2,minmax(0,1fr)); }
        [data-testid="stChatInputContainer"] { padding:.55rem .65rem .75rem !important; }
        .source-line { font-size:.86rem; gap:.2rem .4rem; }
        .source-meta { white-space:normal; }
    }
    </style>
    """, unsafe_allow_html=True)

def sources_inline_html(sources):
    """Render PDF/web sources as a clean left-aligned citation block."""
    if not sources:
        return ""

    lines = [
        '<div class="sources-inline">',
        '<div class="sources-title">📚 Sources used</div>'
    ]

    for source in sources:
        if source.get("type") == "web":
            title = html.escape(
                source.get("title") or "Internet source"
            )
            domain = html.escape(
                source.get("domain") or ""
            )
            link = html.escape(
                source.get("link") or "",
                quote=True
            )
            snippet = html.escape(
                source.get("snippet") or ""
            )

            lines.append(
                '<div class="source-line">'
                '🌐 '
                f'<a href="{link}" target="_blank" rel="noopener noreferrer">'
                f'{title}'
                '</a>'
                f'<span class="source-meta">{domain}</span>'
                + (
                    f'<span class="source-snippet">{snippet}</span>'
                    if snippet else ""
                )
                + '</div>'
            )

        else:
            filename = html.escape(
                source.get("file") or "PDF"
            )
            page = html.escape(
                str(source.get("page") or "")
            )

            lines.append(
                '<div class="source-line">'
                '📄 '
                f'<b>{filename}</b>'
                f'<span class="source-meta">Page {page}</span>'
                '</div>'
            )

    lines.append('</div>')
    return ''.join(lines)

def answer_html(text, language, sources=None):
    rtl = " rtl" if language in ("Urdu", "Arabic") else ""
    lang = "ur" if language == "Urdu" else "ar" if language == "Arabic" else "en"
    return (
        f'<div class="answer-box{rtl}" lang="{lang}">'
        f'{html.escape(str(text or "")).replace(chr(10), "<br>")}'
        f'{sources_inline_html(sources)}</div>'
    )

def render_answer(target, text, language, sources=None):
    target.markdown(answer_html(text, language, sources), unsafe_allow_html=True)

def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS chat_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            message TEXT,
            sources TEXT)"""
        )
        conn.commit()
        conn.close()
        return True

    except sqlite3.Error as exc:
        st.error(f"Database initialization failed: {exc}")
        return False

def save_message(role, message, sources=None):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO chat_history(role,message,sources) VALUES(?,?,?)",
        (
            role,
            str(message),
            json.dumps(sources or [], ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()

def load_messages():
    try:
        conn = sqlite3.connect(DB_FILE)
        rows = conn.execute(
            "SELECT role,message,sources FROM chat_history ORDER BY id"
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return []
    result = []
    for role, message, sources in rows:
        try:
            parsed = json.loads(sources or "[]")
        except Exception:
            parsed = []
        result.append({
            "role": role,
            "content": message,
            "sources": parsed,
        })
    return result

def clear_history():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()

def file_signature(files):
    parts = []
    for f in sorted(files, key=lambda x: x.name):
        data = f.getvalue()
        parts.append(
            f"{f.name}:{len(data)}:{hashlib.sha1(data).hexdigest()}"
        )
    return hashlib.sha1("|".join(parts).encode()).hexdigest()

def remove_index_only():
    for filename in (CHUNKS_FILE, PDF_INFO_FILE):
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError:
                pass
    if os.path.exists(FAISS_FOLDER):
        shutil.rmtree(FAISS_FOLDER, ignore_errors=True)

def remove_all_pdfs():
    remove_index_only()
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)

def save_uploads(files):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    paths = []
    for f in files:
        name = os.path.basename(f.name).replace("..", "_")
        path = os.path.join(UPLOAD_FOLDER, name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        paths.append(path)
    return paths

def process_pdfs(files):
    """Process newly uploaded PDFs and ADD them to whatever is already
    indexed, instead of wiping and rebuilding from only the files that
    happen to be in the uploader widget right now."""
    existing_chunks = st.session_state.get("chunks") or []
    existing_info = st.session_state.get("pdf_info") or []

    if not existing_chunks and os.path.exists(CHUNKS_FILE) and os.path.exists(PDF_INFO_FILE):
        try:
            with open(CHUNKS_FILE, "rb") as f:
                existing_chunks = pickle.load(f)
            with open(PDF_INFO_FILE, "rb") as f:
                existing_info = pickle.load(f)
        except Exception:
            existing_chunks, existing_info = [], []

    existing_names = {item["name"] for item in existing_info}
    new_files = [
        f for f in files
        if os.path.basename(f.name) not in existing_names
    ]

    if not new_files:
        # Nothing new to add — keep the existing index exactly as-is.
        vector_db = st.session_state.get("vector_db")
        if vector_db is None and existing_chunks:
            vector_db, _, _, _ = load_saved_index()
        return (
            vector_db,
            build_source_bm25(existing_chunks, k=RETRIEVAL_K) if existing_chunks else None,
            existing_chunks,
            existing_info,
        )

    paths = save_uploads(new_files)
    documents = []
    new_info = []

    for path in paths:
        name = os.path.basename(path)
        try:
            pages = PyPDFLoader(path).load()
        except Exception as exc:
            st.warning(f"Could not read {name}: {exc}")
            continue
        if not pages:
            continue
        documents.extend(pages)
        new_info.append({
            "name": name,
            "pages": len(pages),
            "chunks": 0,
        })

    if not documents:
        vector_db = st.session_state.get("vector_db")
        if vector_db is None and existing_chunks:
            vector_db, _, _, _ = load_saved_index()
        return (
            vector_db,
            build_source_bm25(existing_chunks, k=RETRIEVAL_K) if existing_chunks else None,
            existing_chunks,
            existing_info,
        )

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        new_chunks = splitter.split_documents(documents)

    except Exception as exc:
        st.error(f"Error while splitting PDF text: {exc}")
        return (
            st.session_state.get("vector_db"),
            st.session_state.get("source_bm25"),
            existing_chunks,
            existing_info,
    )

    if not new_chunks:
        st.warning("No readable text was found in the uploaded PDF.")
        return (
            st.session_state.get("vector_db"),
            st.session_state.get("source_bm25"),
            existing_chunks,
            existing_info,
        )

    for chunk in new_chunks:
        source = os.path.basename(
            str(chunk.metadata.get("source", "")).replace("\\", "/")
        )
        for pdf in new_info:
            if pdf["name"] == source:
                pdf["chunks"] += 1

    all_chunks = existing_chunks + new_chunks
    all_info = existing_info + new_info

    try:
        embedding = load_embedding()

        if embedding is None:
            raise RuntimeError("Embedding model could not be loaded.")

    except Exception as exc:
        st.error(f"Error while loading embedding model: {exc}")
        return (
            st.session_state.get("vector_db"),
            st.session_state.get("source_bm25"),
            existing_chunks,
            existing_info,
        )

    vector_db = st.session_state.get("vector_db")

    if (
        vector_db is None
        and os.path.exists(os.path.join(FAISS_FOLDER, "index.faiss"))
    ):
        try:
            vector_db = FAISS.load_local(
                FAISS_FOLDER,
                embedding,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            st.warning(f"Could not load existing FAISS index: {exc}")
            vector_db = None

    try:
        if vector_db is not None:
            vector_db.add_documents(new_chunks)
        else:
            vector_db = FAISS.from_documents(
                all_chunks,
                embedding,
            )

    except Exception as exc:
        st.error(f"Error while creating PDF search index: {exc}")
        return (
            st.session_state.get("vector_db"),
            st.session_state.get("source_bm25"),
            existing_chunks,
            existing_info,
        )

    try:
        os.makedirs(FAISS_FOLDER, exist_ok=True)

        vector_db.save_local(FAISS_FOLDER)

        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(all_chunks, f)

        with open(PDF_INFO_FILE, "wb") as f:
            pickle.dump(all_info, f)

    except (OSError, pickle.PickleError) as exc:
        st.error(f"Error while saving PDF search index: {exc}")
        return (
            vector_db,
            build_source_bm25(all_chunks, k=RETRIEVAL_K),
            all_chunks,
            all_info,
        )
    except Exception as exc:
        st.error(f"Unexpected error while saving PDF index: {exc}")
        return (
            vector_db,
            build_source_bm25(all_chunks, k=RETRIEVAL_K),
            all_chunks,
            all_info,
        )

    try:
        source_bm25 = build_source_bm25(
            all_chunks,
            k=RETRIEVAL_K,
        )
    except Exception as exc:
        st.error(f"Error while creating BM25 search index: {exc}")
        return (
            vector_db,
            None,
            all_chunks,
            all_info,
        )

    return vector_db, source_bm25, all_chunks, all_info

def load_saved_index():
    required_files = (
        os.path.join(FAISS_FOLDER, "index.faiss"),
        CHUNKS_FILE,
        PDF_INFO_FILE,
    )

    # No saved index exists yet.
    if not all(os.path.exists(path) for path in required_files):
        return None, None, None, []

    try:
        # Load embedding model.
        embedding = load_embedding()

        if embedding is None:
            st.warning(
                "Saved PDF index exists, but the embedding model "
                "could not be loaded."
            )
            return None, None, None, []

        # Load FAISS database.
        vector_db = FAISS.load_local(
            FAISS_FOLDER,
            embedding,
            allow_dangerous_deserialization=True,
        )

        if vector_db is None:
            st.warning("Saved FAISS database could not be loaded.")
            return None, None, None, []

    except Exception as exc:
        st.warning(
            f"Could not load the saved FAISS index: {exc}"
        )
        return None, None, None, []

    # Load saved chunks.
    try:
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)

        if not isinstance(chunks, list) or not chunks:
            st.warning("Saved PDF chunks are empty or invalid.")
            return None, None, None, []

    except (OSError, pickle.PickleError, EOFError) as exc:
        st.warning(
            f"Could not read saved PDF chunks: {exc}"
        )
        return None, None, None, []

    except Exception as exc:
        st.warning(
            f"Unexpected error while reading PDF chunks: {exc}"
        )
        return None, None, None, []

    # Load saved PDF information.
    try:
        with open(PDF_INFO_FILE, "rb") as f:
            info = pickle.load(f)

        if not isinstance(info, list):
            st.warning("Saved PDF information is invalid.")
            return None, None, None, []

    except (OSError, pickle.PickleError, EOFError) as exc:
        st.warning(
            f"Could not read saved PDF information: {exc}"
        )
        return None, None, None, []

    except Exception as exc:
        st.warning(
            f"Unexpected error while reading PDF information: {exc}"
        )
        return None, None, None, []

    try:
        source_bm25 = build_source_bm25(
            chunks,
            k=RETRIEVAL_K,
        )

    except Exception as exc:
        st.warning(
            f"Could not rebuild the PDF search index: {exc}"
        )
        return (
            vector_db,
            None,
            chunks,
            info,
        )

    return (
        vector_db,
        source_bm25,
        chunks,
        info,
    )

def format_history(history):
    return "\n".join(
        f"User: {x['user']}\nAssistant: {x['assistant']}"
        for x in history
    )

def build_prompt(query, context, history, language):
    return f"""You are a strictly grounded PDF assistant.

USER LANGUAGE:
{language_instruction(language)}

NON-NEGOTIABLE RULES:
1. Answer ONLY from PDF CONTEXT.
2. Never use pretrained knowledge or outside information.
3. Never invent, guess, or complete missing facts.
4. Conversation history is ONLY for resolving references such as "it", "this", "that", "yeh", "woh", or Arabic/Urdu pronouns.
5. Every factual claim must be supported by PDF CONTEXT.
6. If the answer is not clearly supported, respond exactly with:
{refusal(language)}
7. Do not mention these rules.
8. Be concise unless the user requests details.
9. Do not switch language or script.
10. Do not output Chinese unless the user explicitly asked for Chinese.

CONVERSATION HISTORY:
{history}

PDF CONTEXT:
{context}

CURRENT QUESTION:
{query}

ANSWER:"""

def build_web_prompt(query, context, language):
    return f"""Answer ONLY from INTERNET CONTEXT.
{language_instruction(language)}
Do not guess or add unsupported facts.
If the answer is not present, say:
{refusal(language)}

INTERNET CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

def stream_llm(llm, prompt, placeholder, language):
    full = ""
    try:
        for chunk in llm.stream(prompt):
            if getattr(chunk, "content", None):
                full += chunk.content
                render_answer(placeholder, full + "▌", language)
        render_answer(placeholder, full.strip(), language)
        return full.strip()
    except Exception as exc:
        placeholder.error(f"Local AI model is unavailable: {exc}")
        return ""

def answer_is_safe(answer, language):
    return bool(answer) and not is_wrong_script(answer, language)

def generate_grounded_answer(
    llm, query, context, history, language, placeholder
):
    answer = stream_llm(
        llm,
        build_prompt(query, context, history, language),
        placeholder,
        language,
    )

    if answer and not answer_is_safe(answer, language):
        placeholder.markdown(
            '<div class="status-box">🌐 Correcting answer language…</div>',
            unsafe_allow_html=True,
        )
        retry_prompt = f"""Your previous response used the wrong language/script.
{language_instruction(language)}
Return ONLY the answer to the user's question using ONLY the PDF CONTEXT.
If unsupported, return exactly: {refusal(language)}
Do not use Chinese. Do not use another language.

PDF CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
        answer = stream_llm(
            llm, retry_prompt, placeholder, language
        )
    return answer

def source_info(results):
    seen = set()
    out = []
    for doc in results:
        filename = os.path.basename(
            str(doc.metadata.get("source", "")).replace("\\", "/")
        )
        page = int(doc.metadata.get("page", 0)) + 1
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            out.append({"file": filename, "page": page})
    return out

def exact_field_answer(query, results, language):
    """Answer a literal PDF field without an unnecessary slow LLM call."""
    if "iqama" not in normalize_text(query).lower():
        return ""
    for doc in results:
        match = re.search(
            r"(?:iqama\s*status|iqama)\s*[:\-]?\s*([^\n.]{2,80})",
            doc.page_content,
            re.IGNORECASE,
        )
        if match:
            value = match.group(1).strip(" :-.")
            if value:
                labels = {
                    "English": "Iqama Status", "Roman Urdu": "Iqama status",
                    "Urdu": "اقامہ اسٹیٹس", "Arabic": "حالة الإقامة",
                }
                return f"{labels.get(language, 'Iqama Status')}: {value}."
    return ""

def render_pdf_gallery(pdf_info):
    """Premium horizontal document rack, with no external image dependency."""
    if not pdf_info:
        return
    tiles = []
    for pdf in pdf_info:
        name = html.escape(pdf["name"])
        tiles.append(
            '<div class="pdf-tile"><div class="pdf-icon">📄</div>'
            f'<div class="pdf-tile-name" title="{name}">{name}</div>'
            f'<div class="pdf-tile-meta">{pdf["pages"]} pages · {pdf["chunks"]} chunks</div></div>'
        )
    st.markdown('<div class="pdf-rack">' + ''.join(tiles) + '</div>', unsafe_allow_html=True)

def render_uploaded_files(files):
    """Always-visible selected-file labels, independent of Streamlit theme CSS."""
    if not files:
        return
    labels = ''.join(
        f'<span class="uploaded-file">✓ {html.escape(item.name)} · {len(item.getvalue()) // 1024} KB</span>'
        for item in files
    )
    st.markdown('<div class="uploaded-files">' + labels + '</div>', unsafe_allow_html=True)

def web_search(query, max_results=4):
    try:
        with DDGS() as ddgs:
            results = []
            for item in ddgs.text(query, max_results=max_results):
                link = str(item.get("href", "")).strip()
                parsed = urlparse(link)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    continue
                results.append({
                    "title": str(item.get("title", "")).strip() or parsed.netloc,
                    "body": str(item.get("body", "")).strip(),
                    "link": link,
                    "domain": parsed.netloc,
                })
            return results
    except Exception:
        return []

def internet_fallback(query, language, target, status):
    """Use the web only after the PDF route is unsupported."""
    status.markdown(
        '<div class="status-box">🌐 Searching Internet sources…</div>',
        unsafe_allow_html=True,
    )
    web_results = web_search(query)
    if not web_results:
        return refusal(language), []
    web_context = "\n\n".join(
        f"Title: {item['title']}\nContent: {item['body']}" for item in web_results
    )
    answer = stream_llm(load_llm(), build_web_prompt(query, web_context, language), target, language)
    if not answer or not answer_is_safe(answer, language):
        return refusal(language), []
    sources = [{
        "type": "web", "title": item["title"], "domain": item["domain"],
        "link": item["link"], "snippet": item["body"][:220],
    } for item in web_results]
    return answer, sources

def handle_query(query):
    
    query = normalize_text(query)
    if not query:
        return

    if "vector_db" not in st.session_state:
        language = detect_language(query)

        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "sources": [],
        })
        save_message("user", query, [])

        with st.chat_message("user", avatar=USER_AVATAR):
            render_answer(
                st.empty(),
                query,
                language,
            )

        answer = refusal(language)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            render_answer(
                st.empty(),
                answer,
                language,
                [],
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": [],
        })
        save_message("assistant", answer, [])
        st.session_state.memory.add_message(
            query,
            answer,
        )

        return

    language = detect_language(query)

    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "sources": [],
    })
    save_message("user", query, [])

    with st.chat_message("user", avatar=USER_AVATAR):
        render_answer(st.empty(), query, language)

    history = st.session_state.memory.get_history()
    start = time.time()

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        status = st.empty()
        status.markdown(
            '<div class="status-box">🔎 Searching your PDFs…</div>',
            unsafe_allow_html=True,
        )

        try:
            rewritten, results = retrieve_documents(
                query,
                st.session_state.vector_db,
                st.session_state.source_bm25,
                st.session_state.chunks,
                history,
                retrieval_k=RETRIEVAL_K,
                final_k=FINAL_K,
            )
        except Exception as exc:
            status.error(f"PDF search failed: {exc}")
            return query,

        sources = []

        if not results:
            target = st.empty()
            answer = refusal(language)
            sources = []

            if st.session_state.answer_mode == "🌐 PDF + Internet":
                answer, sources = internet_fallback(query, language, target, status)

            render_answer(target, answer, language, sources)
            status.empty()
        else:
            status.markdown(
                '<div class="status-box">'
                '🧠 Reading relevant PDF context and preparing the answer…'
                '</div>',
                unsafe_allow_html=True,
            )

            context = compress_context(rewritten, results, max_chars=3000)
            target = st.empty()

            answer = exact_field_answer(query, results, language)

            if answer:
                results = results[:1]
            else:
                try:
                    llm = load_llm()

                    answer = generate_grounded_answer(
                        llm,
                        query,
                        context,
                        format_history(history),
                        language,
                        target,
                    )

                except Exception as exc:
                    st.error(f"AI model error: {exc}")
                    answer = refusal(language)

            status.empty()

            validation = validate_answer(
                query, context, answer
            )

            if (
                not answer
                or is_refusal(answer)
                or validation == "NOT_SUPPORTED"
            ):
                answer = refusal(language)
                sources = []

                if st.session_state.answer_mode == "🌐 PDF + Internet":
                    answer, sources = internet_fallback(
                        query,
                        language,
                        target,
                        status,
                    )

                render_answer(
                    target,
                    answer,
                    language,
                    sources,
                )
                render_read_aloud(
                    answer,
                    language,
                    unique_id=f"live_{time.time_ns()}",
                )

            else:
                if validation == "PARTIALLY_SUPPORTED":
                    partial = {
                        "English": (
                            "I can only partially answer this from the PDF.\n\n"
                        ),
                        "Urdu": (
                            "میں PDF میں موجود معلومات کی بنیاد پر صرف جزوی جواب دے سکتا ہوں۔\n\n"
                        ),
                        "Roman Urdu": (
                            "Main PDF mein mojood maloomat ki bunyaad par sirf kuch hissa bata sakta hoon.\n\n"
                        ),
                        "Arabic": (
                            "يمكنني الإجابة جزئيًا فقط بناءً على المعلومات الموجودة في ملف PDF.\n\n"
                        ),
                    }[language]

                    answer = partial + answer

                if not answer_is_safe(answer, language):
                    answer = refusal(language)
                    sources = []
                else:
                    sources = source_info(results)

                render_answer(
                    target,
                    answer,
                    language,
                    sources,
                )
                render_read_aloud(
                    answer,
                    language,
                    unique_id=f"live_{time.time_ns()}",
                )
                        
    
        st.caption(f"⏱️ {time.time() - start:.1f}s")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    save_message("assistant", answer, sources)
    st.session_state.memory.add_message(query, answer)

@st.cache_resource(show_spinner="🎙️ Loading Whisper voice model…")
def load_whisper():
    return whisper.load_model(
        os.getenv("AI_PDF_WHISPER_MODEL", "base")
    )

def transcribe_audio(data):
    path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav"
        ) as file:
            file.write(data)
            path = file.name
        result = load_whisper().transcribe(
            path,
            task="transcribe",
            fp16=False,
            temperature=0,
            condition_on_previous_text=False,
        )

        text = normalize_text(result.get("text", ""))
        if not text:
            return ""

        # Silence / hallucination guard.
        # Whisper reports per-segment `no_speech_prob` (confidence that a
        # segment contains no speech), `avg_logprob` (confidence in the
        # decoded tokens), and `compression_ratio` (how repetitive the
        # decoded text is — this is the same statistic Whisper's own
        # decoder uses internally to flag a failed/looping decode, with
        # 2.4 being Whisper's own default threshold). On silent audio,
        # Whisper sometimes doesn't just emit nothing — it can loop a
        # short phrase (often drawn straight from initial_prompt) over
        # and over, which looks like real text but is highly repetitive
        # and is caught by the compression_ratio check even when
        # no_speech_prob/avg_logprob alone don't flag it.
        segments = result.get("segments") or []

        def segment_is_bad(seg):
            silence_like = (
                seg.get("no_speech_prob", 0.0) > 0.6
                and seg.get("avg_logprob", 0.0) < -1.0
            )
            repetitive = seg.get("compression_ratio", 0.0) > 2.4
            return silence_like or repetitive

        if segments and all(segment_is_bad(seg) for seg in segments):
            return ""

        return text
    except Exception as exc:
        st.error(f"Voice transcription failed: {exc}")
        return ""
    finally:
        if path and os.path.exists(path):
            os.remove(path)
def text_to_speech(text, language):
    voices = {
        "English": "en-US-AriaNeural",
        "Urdu": "ur-PK-UzmaNeural",
        "Arabic": "ar-SA-ZariyahNeural",
        "Roman Urdu": "ur-PK-UzmaNeural",
    }

    voice = voices.get(
        language,
        "en-US-AriaNeural",
    )

    text = str(text).strip()

    if not text:
        return None

    filename = (
        "ai_pdf_tts_"
        + hashlib.sha1(
            text.encode("utf-8")
        ).hexdigest()
        + ".mp3"
    )

    path = os.path.join(
        tempfile.gettempdir(),
        filename,
    )

    try:
        import asyncio

        async def generate():
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate="+0%",
                volume="+0%",
            )

            await communicate.save(path)

        # Run Edge TTS
        asyncio.run(generate())

        if not os.path.isfile(path):
            st.error("❌ Edge TTS did not create the audio file.")
            return None

        if os.path.getsize(path) == 0:
            st.error("❌ Edge TTS created an empty audio file.")
            return None

        return path

    except Exception as exc:
        st.error(
            f"❌ Read aloud failed: {exc}"
        )
        return None
    
def render_read_aloud(text, language, unique_id):
    if not text:
        return

    text = str(text).strip()

    text_hash = hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()

    button_key = f"read_aloud_{unique_id}_{text_hash[:12]}"
    audio_key = f"audio_{unique_id}_{text_hash[:12]}"
    # Audio already generated
    if audio_key in st.session_state:
        audio_data = st.session_state[audio_key]

        if audio_data:
            st.audio(
                audio_data,
                format="audio/mp3",
                autoplay=False,
            )

        return

    # Generate on first click
    if st.button(
        "🔊 Read aloud",
        key=button_key,
        help="Listen to this answer",
    ):
        with st.spinner("🔊 Generating audio..."):
            audio_path = text_to_speech(
                text,
                language,
            )

        if not audio_path:
            st.error("❌ Failed to generate audio.")
            return

        try:
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()

            if not audio_data:
                st.error("❌ Generated audio is empty.")
                return

            # Save audio
            st.session_state[audio_key] = audio_data

            # Show audio immediately
            st.audio(
                audio_data,
                format="audio/mp3",
                autoplay=False,
            )

            # Remove temp file
            try:
                os.remove(audio_path)
            except OSError:
                pass

        except Exception as exc:
            st.error(
                f"❌ Could not load generated audio: {exc}"
            )
            
try:
    init_db()
except sqlite3.Error as exc:
    st.error(f"Database initialization failed: {exc}")
except Exception as exc:
    st.error(f"Unexpected database error: {exc}")

if "messages" not in st.session_state:
    st.session_state.messages = load_messages()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
    pairs = zip(
        st.session_state.messages[0::2],
        st.session_state.messages[1::2],
    )
    for user_msg, assistant_msg in pairs:
        if (
            user_msg["role"] == "user"
            and assistant_msg["role"] == "assistant"
        ):
            st.session_state.memory.add_message(
                user_msg["content"],
                assistant_msg["content"],
            )

if "answer_mode" not in st.session_state:
    st.session_state.answer_mode = "📄 PDF Only"
if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None

inject_css()

try:

    with open(HERO_LOGO_PATH, "rb") as logo_file:
        _hero_logo_b64 = base64.b64encode(
            logo_file.read()
        ).decode()

    logo_html = (
        f'<span class="bot-mark">'
        f'<img src="data:image/png;base64,{_hero_logo_b64}" alt="bot"/>'
        f'</span>'
    )

except (FileNotFoundError, OSError):
    _hero_logo_b64 = ""
    logo_html = '<span class="bot-mark">🤖</span>'

st.markdown(
    '<div class="hero">'
    '<div>'
    '<div class="hero-title">'
    f'{logo_html}'
    'AI PDF Chatbot'
    '</div>'
    '<div class="hero-sub">Ask anything from your PDFs</div>'
    '</div>'
    '<div class="hero-badges">'
    '<span class="hero-badge">◉ Powered by RAG</span>'
    '<span class="hero-badge purple">◉ Multi-language</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("⚙️ Controls")
    st.radio(
        "Answer mode",
        ["📄 PDF Only", "🌐 PDF + Internet"],
        key="answer_mode",
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):
        clear_history()
        st.session_state.messages = []
        st.session_state.memory = ConversationMemory()
        st.rerun()

    if st.button("🧹 Remove All PDFs"):
        remove_all_pdfs()
        for key in (
            "vector_db",
            "source_bm25",
            "chunks",
            "pdf_info",
            "upload_signature",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    chat_text = "\n\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in st.session_state.messages
    )
    if chat_text:
        st.download_button(
            "⬇️ Download Chat",
            chat_text,
            "chat_history.txt",
            "text/plain",
        )

st.markdown("### 📄 Upload your PDFs")

uploaded = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="visible",
    key="pdf_uploader",
)
render_uploaded_files(uploaded)

if uploaded:
    signature = file_signature(uploaded)
    if signature != st.session_state.upload_signature:
        with st.spinner(
            "📚 Processing PDFs and building the search index…"
        ):
            db, source_bm25, chunks, info = process_pdfs(
                uploaded
            )

        if db is not None:
            st.session_state.vector_db = db
            st.session_state.source_bm25 = source_bm25
            st.session_state.chunks = chunks
            st.session_state.pdf_info = info
            st.session_state.upload_signature = signature
            st.success(
                f"✅ {len(info)} PDF(s) ready — "
                f"{sum(x['pages'] for x in info)} pages, "
                f"{len(chunks)} chunks."
            )

if "vector_db" not in st.session_state:
    db, source_bm25, chunks, info = load_saved_index()
    if db is not None:
        st.session_state.vector_db = db
        st.session_state.source_bm25 = source_bm25
        st.session_state.chunks = chunks
        st.session_state.pdf_info = info

if st.session_state.get("pdf_info"):
    st.sidebar.markdown("### 📂 Indexed PDFs")
    total_pages = sum(
        item["pages"] for item in st.session_state.pdf_info
    )
    total_chunks = sum(
        item["chunks"] for item in st.session_state.pdf_info
    )

    for pdf in st.session_state.pdf_info:
        st.sidebar.markdown(
            f'<div class="pdf-card">'
            f'<div class="pdf-name">📄 '
            f'{html.escape(pdf["name"])}</div>'
            f'<div class="pdf-meta">'
            f'{pdf["pages"]} pages · {pdf["chunks"]} chunks'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown(
        f'<span class="pill">📚 '
        f'{len(st.session_state.pdf_info)} PDFs</span>'
        f'<span class="pill">📄 {total_pages} pages</span>'
        f'<span class="pill">🧩 {total_chunks} chunks</span>',
        unsafe_allow_html=True,
    )
    render_pdf_gallery(st.session_state.pdf_info)

if "vector_db" not in st.session_state:
    st.markdown(
        '<div class="empty-state">'
        '<b>📄 Upload your PDF above to start.</b><br>'
        'The assistant selects the relevant document before answering.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="chat-heading">💬 Chat</div>',
        unsafe_allow_html=True,
    )


# CHAT HISTORY
for message_index, message in enumerate(
    st.session_state.messages
):

    language = detect_language(
        message["content"]
    )

    with st.chat_message(
        message["role"],
        avatar=(
            BOT_AVATAR
            if message["role"] == "assistant"
            else USER_AVATAR
        ),
    ):
        render_answer(
            st.empty(),
            message["content"],
            language,
            (
                message.get("sources")
                if message["role"] == "assistant"
                else None
            ),
        )

        if message["role"] == "assistant":
            render_read_aloud(
                message["content"],
                language,
                unique_id=message_index,
        )

# CHAT INPUT

try:
    prompt = st.chat_input(
        "Ask about your PDFs, or use the microphone…",
        accept_audio=True,
        audio_sample_rate=16000,
        key="main_chat_input",
    )

except TypeError:
    st.caption(
        "🎙️ Update Streamlit to 1.49+ "
        "to enable the microphone in the chat input."
    )

if prompt:

    audio = getattr(
        prompt,
        "audio",
        None,
    )
    text = getattr(
        prompt,
        "text",
        None,
    )
    if audio is not None:
        with st.status("🎙️ Transcribing your recording…",expanded=False,):
            text = transcribe_audio(
                audio.getvalue()
            )
    if text:
        handle_query(text)
st.markdown(
    '<div class="features">'
    '<div class="feature"><b>🤖 Smart PDF Assistant</b>Answers grounded in your documents</div>'
    '<div class="feature"><b>🌐 Multi-language</b>English, اردو, العربية and Roman Urdu</div>'
    '<div class="feature"><b>🎙️ Voice Input</b>Ask questions with your microphone</div>'
    '<div class="feature"><b>⚡ Fast & Accurate</b>Hybrid PDF search with clear sources</div>'
    '</div>',
    unsafe_allow_html=True,
)