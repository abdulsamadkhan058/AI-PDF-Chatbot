"""Professional Multilingual AI PDF Chatbot -- Chainlit edition.

This is the Chainlit web entry point (`chainlit run app.py`). It reuses the
RAG modules below:

    answer_validator.py     -- deterministic answer validation
    context_compressor.py   -- PDF context compression
    conversation_memory.py  -- rolling chat memory
    language_utils.py       -- language detection / script validation
    retrieval.py            -- hybrid FAISS + BM25 + rerank retrieval
    core/embeddings.py      -- cached multilingual embeddings
    core/llm.py             -- cached Gemini chat model

and the table/chart analysis module:

    table_chart_analysis.py -- table + chart/graph question answering
"""

import asyncio
import hashlib
import html
import io
import json
import os
import pickle
import re
import shutil
import sqlite3
import tempfile
import time
import wave
from urllib.parse import urlparse

import chainlit as cl
import numpy as np
from ddgs import DDGS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)

from answer_validator import validate_answer
from context_compressor import compress_context
from conversation_memory import ConversationMemory
from core.embeddings import load_embedding
from core.llm import load_llm
from core.llm import get_api_key as core_llm_get_api_key
from language_utils import (
    detect_language,
    is_wrong_script,
    language_instruction,
    normalize_text,
)
from retrieval import build_source_bm25, retrieve_documents
import table_chart_analysis as tca
import os
import subprocess
import streamlit as st

# ---------------------------------------------------------------------------
# Constants (unchanged from the Streamlit app so existing saved indexes /
# chat history / uploaded PDFs keep working with zero migration).
# ---------------------------------------------------------------------------

APP_TITLE = "AI PDF Chatbot"
FAISS_FOLDER = os.path.join(BASE_DIR, "faiss_db")
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.pkl")
PDF_INFO_FILE = os.path.join(BASE_DIR, "pdf_info.pkl")
INDEX_META_FILE = os.path.join(BASE_DIR, "index_meta.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_FILE = os.path.join(BASE_DIR, "chat_history.db")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_K = 6
FINAL_K = 3
# The browser's mic recorder (Chainlit's WavRecorder, confirmed by reading
# its shipped frontend bundle) ALWAYS captures at 24000 Hz — the
# `sample_rate` in .chainlit/config.toml is not actually wired to it. Writing
# the recorded PCM into a WAV file at any other rate here would play it back
# at the wrong speed/pitch, which is what was garbling Whisper's results
# (wrong language, hallucinated text) — this MUST stay 24000.
WHISPER_SAMPLE_RATE = 24000
GEMINI_MODEL = os.getenv("AI_PDF_GEMINI_MODEL", "gemini-3.6-flash").strip() or "gemini-3.6-flash"

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


# ---------------------------------------------------------------------------
# SQLite chat history (same schema/behaviour as the Streamlit app)
# ---------------------------------------------------------------------------

def _session_id():
    """Return the current Chainlit session id for chat-history isolation."""
    return str(cl.user_session.get("id") or "anonymous")


def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS chat_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT,
            message TEXT,
            sources TEXT)"""
        )
        # Migrate older local databases created before session isolation.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chat_history)")}
        if "session_id" not in columns:
            conn.execute("ALTER TABLE chat_history ADD COLUMN session_id TEXT")
            conn.execute(
                "UPDATE chat_history SET session_id = 'legacy' WHERE session_id IS NULL"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_history_session "
            "ON chat_history(session_id, id)"
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as exc:
        print(f"[db] init failed: {exc}")
        return False


def save_message(role, message, sources=None):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute(
            "INSERT INTO chat_history(session_id,role,message,sources) VALUES(?,?,?,?)",
            (_session_id(), role, str(message), json.dumps(sources or [], ensure_ascii=False)),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as exc:
        print(f"[db] save failed: {exc}")


def load_messages():
    try:
        conn = sqlite3.connect(DB_FILE)
        rows = conn.execute(
            "SELECT role,message,sources FROM chat_history "
            "WHERE session_id = ? ORDER BY id",
            (_session_id(),),
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
        result.append({"role": role, "content": message, "sources": parsed})
    return result


def clear_history():
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (_session_id(),))
        conn.commit()
        conn.close()
    except sqlite3.Error as exc:
        print(f"[db] clear failed: {exc}")


# ---------------------------------------------------------------------------
# PDF upload / indexing (adapted for Chainlit, without Streamlit UI calls)
# ---------------------------------------------------------------------------

class _Upload:
    """Tiny adapter so process_pdfs() doesn't care whether a file came
    from a Chainlit message element or somewhere else."""

    def __init__(self, name, path):
        self.name = name
        self.path = path


def file_signature(uploads):
    parts = []
    for f in sorted(uploads, key=lambda x: x.name):
        with open(f.path, "rb") as fh:
            data = fh.read()
        parts.append(f"{f.name}:{len(data)}:{hashlib.sha1(data).hexdigest()}")
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


def save_uploads(uploads):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    paths = []
    for f in uploads:
        name = os.path.basename(f.name).replace("..", "_")
        dest = os.path.join(UPLOAD_FOLDER, name)
        shutil.copyfile(f.path, dest)
        paths.append(dest)
    return paths


def process_pdfs(uploads, log):
    """Process newly uploaded PDFs and ADD them to whatever is already
    indexed. `log(msg)` is called with human-readable progress/errors so
    the caller can surface them as a Chainlit Step."""
    existing_chunks = cl.user_session.get("chunks") or []
    existing_info = cl.user_session.get("pdf_info") or []

    if not existing_chunks and os.path.exists(CHUNKS_FILE) and os.path.exists(PDF_INFO_FILE):
        try:
            with open(CHUNKS_FILE, "rb") as f:
                existing_chunks = pickle.load(f)
            with open(PDF_INFO_FILE, "rb") as f:
                existing_info = pickle.load(f)
        except Exception:
            existing_chunks, existing_info = [], []

    existing_names = {item["name"] for item in existing_info}
    new_files = [f for f in uploads if os.path.basename(f.name) not in existing_names]

    if not new_files:
        vector_db = cl.user_session.get("vector_db")
        if vector_db is None and existing_chunks:
            vector_db, _, _, _ = load_saved_index(log)
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
            log(f"⚠️ Could not read {name}: {exc}")
            continue
        if not pages:
            continue

        # Scanned/image-only PDFs extract as empty (or near-empty) text per
        # page. Rather than silently indexing nothing for those pages, fall
        # back to reading them with Gemini vision (reusing the same call
        # already used for charts — no extra OCR dependency needed).
        ocr_count = 0
        for doc in pages:
            if len(doc.page_content.strip()) >= 20:
                continue
            page_number = int(doc.metadata.get("page", 0)) + 1
            ocr_text = tca.ocr_scanned_page(path, page_number, gemini_vision_call)
            if ocr_text:
                doc.page_content = ocr_text
                ocr_count += 1
        if ocr_count:
            log(f"🔎 OCR-recovered text on {ocr_count} scanned page(s) in {name}.")

        documents.extend(pages)
        new_info.append({"name": name, "pages": len(pages), "chunks": 0, "path": path})

    if not documents:
        vector_db = cl.user_session.get("vector_db")
        if vector_db is None and existing_chunks:
            vector_db, _, _, _ = load_saved_index(log)
        return (
            vector_db,
            build_source_bm25(existing_chunks, k=RETRIEVAL_K) if existing_chunks else None,
            existing_chunks,
            existing_info,
        )

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        new_chunks = splitter.split_documents(documents)
    except Exception as exc:
        log(f"❌ Error while splitting PDF text: {exc}")
        return (
            cl.user_session.get("vector_db"),
            cl.user_session.get("source_bm25"),
            existing_chunks,
            existing_info,
        )

    if not new_chunks:
        log("⚠️ No readable text was found in the uploaded PDF.")
        return (
            cl.user_session.get("vector_db"),
            cl.user_session.get("source_bm25"),
            existing_chunks,
            existing_info,
        )

    for chunk in new_chunks:
        source = os.path.basename(str(chunk.metadata.get("source", "")).replace("\\", "/"))
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
        log(f"❌ Error while loading embedding model: {exc}")
        return (
            cl.user_session.get("vector_db"),
            cl.user_session.get("source_bm25"),
            existing_chunks,
            existing_info,
        )

    vector_db = cl.user_session.get("vector_db")
    if vector_db is None and os.path.exists(os.path.join(FAISS_FOLDER, "index.faiss")):
        try:
            vector_db = FAISS.load_local(
                FAISS_FOLDER, embedding, allow_dangerous_deserialization=True
            )
        except Exception as exc:
            log(f"⚠️ Could not load existing FAISS index: {exc}")
            vector_db = None

    try:
        if vector_db is not None:
            vector_db.add_documents(new_chunks)
        else:
            vector_db = FAISS.from_documents(all_chunks, embedding)
    except Exception as exc:
        log(f"❌ Error while creating PDF search index: {exc}")
        return (
            cl.user_session.get("vector_db"),
            cl.user_session.get("source_bm25"),
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
    except Exception as exc:
        log(f"❌ Error while saving PDF index: {exc}")
        return (
            vector_db,
            build_source_bm25(all_chunks, k=RETRIEVAL_K),
            all_chunks,
            all_info,
        )

    try:
        source_bm25 = build_source_bm25(all_chunks, k=RETRIEVAL_K)
    except Exception as exc:
        log(f"❌ Error while creating BM25 search index: {exc}")
        return vector_db, None, all_chunks, all_info

    return vector_db, source_bm25, all_chunks, all_info


def load_saved_index(log=print):
    required_files = (os.path.join(FAISS_FOLDER, "index.faiss"), CHUNKS_FILE, PDF_INFO_FILE)
    if not all(os.path.exists(path) for path in required_files):
        return None, None, None, []

    try:
        embedding = load_embedding()
        if embedding is None:
            log("⚠️ Saved PDF index exists, but the embedding model could not be loaded.")
            return None, None, None, []
        vector_db = FAISS.load_local(
            FAISS_FOLDER, embedding, allow_dangerous_deserialization=True
        )

        # Guard against a stale/mismatched FAISS index. If the number of
        # vectors does not match the saved PDF chunks, rebuild the index
        # from the authoritative chunks.pkl instead of silently searching
        # an incompatible index.
        try:
            with open(CHUNKS_FILE, "rb") as f:
                saved_chunks = pickle.load(f)
            vector_count = int(getattr(vector_db.index, "ntotal", -1))
            if isinstance(saved_chunks, list) and saved_chunks and vector_count != len(saved_chunks):
                vector_db = FAISS.from_documents(saved_chunks, embedding)
                os.makedirs(FAISS_FOLDER, exist_ok=True)
                vector_db.save_local(FAISS_FOLDER)
        except Exception as exc:
            log(f"⚠️ Could not verify saved FAISS index; continuing with loaded index: {exc}")
    except Exception as exc:
        log(f"⚠️ Could not load the saved FAISS index: {exc}")
        return None, None, None, []

    try:
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        if not isinstance(chunks, list) or not chunks:
            return None, None, None, []
    except Exception as exc:
        log(f"⚠️ Could not read saved PDF chunks: {exc}")
        return None, None, None, []

    try:
        with open(PDF_INFO_FILE, "rb") as f:
            info = pickle.load(f)
        if not isinstance(info, list):
            return None, None, None, []
    except Exception as exc:
        log(f"⚠️ Could not read saved PDF information: {exc}")
        return None, None, None, []

    try:
        source_bm25 = build_source_bm25(chunks, k=RETRIEVAL_K)
    except Exception as exc:
        log(f"⚠️ Could not rebuild the PDF search index: {exc}")
        return vector_db, None, chunks, info

    return vector_db, source_bm25, chunks, info


# ---------------------------------------------------------------------------
# Prompting + generation (adapted for the Chainlit application)
# ---------------------------------------------------------------------------

def format_history(history):
    return "\n".join(f"User: {x['user']}\nAssistant: {x['assistant']}" for x in history)


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
6. IMPORTANT: If PDF CONTEXT contains the answer, answer it directly even when the user's wording, spelling, language, or phrasing differs from the PDF.
7. Refuse ONLY when the PDF CONTEXT genuinely does not contain enough information to answer. In that case respond exactly with:
{refusal(language)}
8. Do not mention these rules.
9. Be concise unless the user requests details.
10. Do not switch language or script.
11. Do not output Chinese unless the user explicitly asked for Chinese.

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


class LLMGenerationError(RuntimeError):
    """Raised when the answer model cannot produce a usable response."""


def _content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def call_llm(llm, prompt):
    """Call the configured model and fail loudly enough for the UI to diagnose it."""
    try:
        result = llm.invoke(prompt)
        text = _content_to_text(getattr(result, "content", result)).strip()
        if not text:
            raise LLMGenerationError("The Gemini model returned an empty response.")
        return text
    except LLMGenerationError:
        raise
    except Exception as exc:
        print(f"[llm] error: {type(exc).__name__}: {exc}")
        raise LLMGenerationError(
            f"Gemini request failed ({type(exc).__name__}): {exc}"
        ) from exc


def answer_is_safe(answer, language):
    return bool(answer) and not is_wrong_script(answer, language)


def generate_grounded_answer(llm, query, context, history, language):
    answer = call_llm(llm, build_prompt(query, context, history, language))
    if answer and not answer_is_safe(answer, language):
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
        answer = call_llm(llm, retry_prompt)
    return answer


_WORD_RE = re.compile(r"[a-zA-Z\u0600-\u06FF]{3,}")


def _only_used_sources(answer, results):
    """Retrieval brings several chunks into context, but the model doesn't
    always draw from all of them. Only keep chunks whose distinctive words
    actually show up in the generated answer, so 'Sources used' reflects
    what was really cited instead of every page that was merely retrieved."""
    if len(results) <= 1:
        return results
    answer_words = set(_WORD_RE.findall((answer or "").lower()))
    if not answer_words:
        return results
    kept = []
    for doc in results:
        doc_words = set(_WORD_RE.findall((doc.page_content or "").lower()))
        if not doc_words:
            continue
        overlap = len(answer_words & doc_words) / len(doc_words)
        if overlap >= 0.12:
            kept.append(doc)
    # Never end up with an empty source list — fall back to everything
    # retrieved if the heuristic is too strict for this particular answer.
    return kept or results


def source_info(results):
    seen = set()
    out = []
    for doc in results:
        filename = os.path.basename(str(doc.metadata.get("source", "")).replace("\\", "/"))
        page = int(doc.metadata.get("page", 0)) + 1
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            out.append({"file": filename, "page": page})
    return out


def _dedupe_sources(sources):
    """Final defensive de-duplication so the rendered source list is always
    clean, regardless of which code path produced it."""
    seen = set()
    out = []
    for source in sources or []:
        if source.get("type") == "web":
            key = ("web", source.get("link") or (source.get("domain"), source.get("title")))
        else:
            key = ("pdf", source.get("file"), str(source.get("page")))
        if key in seen:
            continue
        seen.add(key)
        out.append(source)
    return out


def sources_markdown(sources):
    """Render sources after the actual answer, never as the answer itself."""
    sources = _dedupe_sources(sources)
    if not sources:
        return ""
    lines = ["\n\n---\n**📚 Sources used**"]
    for source in sources:
        if source.get("type") == "web":
            title = source.get("title") or "Internet source"
            domain = source.get("domain") or ""
            link = source.get("link") or ""
            lines.append(f"- 🌐 [{title}]({link}) — `{domain}`")
        else:
            filename = source.get("file") or "PDF"
            page = source.get("page") or ""
            lines.append(f"- 📄 **{filename}** — Page {page}")
    return "\n".join(lines)


def _df_to_markdown(df, max_rows=20):
    """Create a clean Markdown table from an extracted pandas DataFrame."""
    try:
        if df is None or df.empty:
            return ""
        view = df.head(max_rows).copy()
        view.columns = [str(c).replace("|", "\\|").replace("\n", " ") for c in view.columns]
        for col in view.columns:
            view[col] = view[col].map(lambda x: str(x).replace("|", "\\|").replace("\n", " "))
        headers = [str(c) for c in view.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in view.columns) + " |")
        table = "\n".join(lines)
        if len(df) > max_rows:
            table += f"\n\n*Showing first {max_rows} of {len(df)} rows.*"
        return table
    except Exception:
        return ""


def _table_context(tables, max_tables=8, max_rows=60):
    """Serialize extracted tables for an LLM fallback without inventing data."""
    blocks = []
    for item in tables[:max_tables]:
        df = item.get("dataframe")
        if df is None or df.empty:
            continue
        try:
            csv_text = df.head(max_rows).to_csv(index=False)
        except Exception:
            csv_text = str(df.head(max_rows))
        blocks.append(
            f"TABLE page={item.get('page')} table={item.get('table_index', 0)}\n{csv_text}"
        )
    return "\n\n".join(blocks)


def _is_table_refusal(answer, language):
    value = normalize_text(answer).lower()
    refusal_text = normalize_text(getattr(tca, "REFUSAL_NO_TABLE", {}).get(language, "")).lower()
    return not value or (refusal_text and value == refusal_text)


def _looks_like_table_display_request(query):
    q = normalize_text(query).lower()
    hints = (
        "show table", "display table", "give me table", "show data",
        "table data", "list all rows", "all rows", "table dikhao",
        "data dikhao", "table do", "جدول دکھ", "الجدول",
    )
    return any(h in q for h in hints)


def _answer_table_with_llm(query, tables, language):
    """Natural-language fallback for complex table questions.

    This is deliberately grounded only in values extracted by pdfplumber.
    If extraction produced no table, the caller falls back to normal RAG.
    """
    table_context = _table_context(tables)
    if not table_context:
        return ""
    prompt = f"""You are answering a question about tables extracted from a PDF.
{language_instruction(language)}
Use ONLY the TABLE DATA below. Never invent, estimate, or use outside knowledge.
If the requested value is not present or cannot be calculated from these rows,
clearly say that it is not available.
When useful, show the result as a Markdown table.

TABLE DATA:
{table_context}

QUESTION:
{query}

ANSWER:"""
    try:
        return call_llm(load_llm(), prompt)
    except Exception as exc:
        print(f"[table-llm] fallback failed: {type(exc).__name__}: {exc}")
        return ""


def web_search(query, max_results=4):
    import time as _time

    last_error = None
    for attempt in range(3):
        try:
            with DDGS(timeout=10) as ddgs:
                results = []
                for item in ddgs.text(
                    query,
                    max_results=max_results,
                    region="wt-wt",
                    safesearch="off",
                ):
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
                if results:
                    seen_links = set()
                    unique_results = []
                    for r in results:
                        if r["link"] in seen_links:
                            continue
                        seen_links.add(r["link"])
                        unique_results.append(r)
                    if unique_results:
                        return unique_results
                last_error = "no results returned"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[web_search] attempt {attempt + 1}/3 failed: {last_error}")
            _time.sleep(1.5 * (attempt + 1))
    print(f"[web_search] giving up after 3 attempts. last_error={last_error}")
    return []


async def internet_fallback(query, language, step):
    step.output = "🌐 Searching Internet sources…"
    await step.update()
    web_results = await cl.make_async(web_search)(query)
    if not web_results:
        return refusal(language), []
    web_context = "\n\n".join(
        f"Title: {item['title']}\nContent: {item['body']}" for item in web_results
    )
    answer = await cl.make_async(call_llm)(load_llm(), build_web_prompt(query, web_context, language))
    if not answer or not answer_is_safe(answer, language):
        return refusal(language), []
    sources = []
    seen_links = set()
    for item in web_results:
        link = item["link"]
        key = link or (item["domain"], item["title"])
        if key in seen_links:
            continue
        seen_links.add(key)
        sources.append({
            "type": "web", "title": item["title"], "domain": item["domain"],
            "link": link, "snippet": item["body"][:220],
        })
    return answer, sources


def _fmt_num(value):
    """Format a number for display without ugly trailing '.0' (e.g. 2023.0 -> 2023)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f.is_integer():
        return str(int(f))
    return f"{f:g}"


def _deterministic_table_summary(df, language):
    """Build a clean, professional per-column summary using the table's
    ACTUAL header names (never generic placeholders like 'col_0')."""
    if df is None or df.empty:
        return ""
    rows, cols = df.shape
    headers = ", ".join(str(c) for c in df.columns)
    intros = {
        "English": f"This table has **{rows} rows** and **{cols} columns**: {headers}.",
        "Roman Urdu": f"Is table mein **{rows} rows** aur **{cols} columns** hain: {headers}.",
        "Urdu": f"اس جدول میں **{rows} قطاریں** اور **{cols} کالم** ہیں: {headers}۔",
        "Arabic": f"يحتوي هذا الجدول على **{rows} صفوف** و **{cols} أعمدة**: {headers}.",
    }
    lines = [intros.get(language, intros["English"])]
    for col in df.columns:
        nums = tca.numeric_series(df[col]).dropna()
        if len(nums) < 2:
            continue
        lines.append(
            f"- **{col}**: min {_fmt_num(nums.min())}, max {_fmt_num(nums.max())}, "
            f"average {_fmt_num(round(float(nums.mean()), 2))}, total {_fmt_num(nums.sum())}."
        )
    return "\n".join(lines)


def _humanize_table_answer(answer):
    """Turn the deterministic aggregate phrasing the table module returns
    (e.g. "Maximum (Salary): 30 — page 2.") into a normal sentence.
    Leaves genuine LLM / markdown-table answers untouched."""
    import re
    if not answer:
        return answer
    pattern = re.compile(
        r"^(Maximum|Minimum|Average|Sum|Total|Count)\s*\(([^)]+)\)\s*:\s*"
        r"([^\u2014\-\n]+?)\s*[\u2014-]\s*page\s*([^\.\n]+)\.?\s*$",
        re.IGNORECASE,
    )
    match = pattern.match(answer.strip())
    if not match:
        return answer
    agg, column, value, page = (g.strip() for g in match.groups())
    phrasing = {
        "maximum": f"The highest value in **{column}** is **{value}** (page {page}).",
        "minimum": f"The lowest value in **{column}** is **{value}** (page {page}).",
        "average": f"The average value in **{column}** is **{value}** (page {page}).",
        "sum": f"The total for **{column}** is **{value}** (page {page}).",
        "total": f"The total for **{column}** is **{value}** (page {page}).",
        "count": f"There are **{value}** entries in **{column}** (page {page}).",
    }
    return phrasing.get(agg.lower(), answer)


def exact_field_answer(query, results, language):
    import re
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


# ---------------------------------------------------------------------------
# Table / chart vision hook (used only by table_chart_analysis.answer_chart_query)
# ---------------------------------------------------------------------------

def gemini_vision_call(image_bytes, prompt):
    """Best-effort multimodal call to Gemini for chart reading. Returns None
    only when the feature is unavailable (no API key configured) — any
    actual API error (quota exhausted, bad request, network issue, etc.) is
    allowed to raise so the caller (answer_chart_query / analyze_all_charts,
    which already catch per-page and surface a friendly message) can tell
    the difference between "genuinely no chart here" and "the request
    failed" instead of both looking identical to the user."""
    # Reuse the SAME key lookup as normal text answers (core.llm._get_api_key),
    # which checks GOOGLE_API_KEY first, then GEMINI_API_KEY. This function
    # previously only checked GEMINI_API_KEY directly, so on any .env that
    # only sets GOOGLE_API_KEY (Google's own documented variable name), every
    # single chart/vision call silently returned nothing — text Q&A worked
    # fine (it already used the shared lookup) while every chart question
    # failed with a generic "could not be reliably extracted" message.
    api_key = core_llm_get_api_key()
    if not api_key:
        return None
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
    )
    return getattr(response, "text", None)


# ---------------------------------------------------------------------------
# Whisper (voice input) + Edge TTS (read aloud)
# ---------------------------------------------------------------------------

_whisper_model = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        # "base" repeatedly misidentified/mis-transcribed Urdu & Arabic speech
        # (landing on Russian/Portuguese/gibberish English) — it's simply too
        # small a model for reliable non-English language ID. "small" is
        # dramatically better at this while still running fine on CPU for
        # short voice queries. Override with AI_PDF_WHISPER_MODEL if needed.
        model_name = os.getenv("AI_PDF_WHISPER_MODEL", "small")
        print(f"🎙️ Loading Whisper voice model ({model_name})…")
        _whisper_model = whisper.load_model(model_name)
    return _whisper_model


# Whisper's own language auto-detection runs across ~99 languages using a
# single 30s window. On short/accented clips (a few spoken words) it can
# lock onto a completely wrong language (observed: Urdu/Arabic speech
# misdetected as Portuguese, producing a hallucinated repeated phrase).
# Restricting the vote to the languages this app actually supports fixes
# that failure mode. Configurable via AI_PDF_VOICE_LANGUAGES (comma-separated
# Whisper language codes), default matches the app's Multi-language support.
VOICE_LANGUAGES = tuple(
    code.strip() for code in os.getenv("AI_PDF_VOICE_LANGUAGES", "en,ur,ar").split(",") if code.strip()
)


def _detect_voice_language(model, path):
    """Language guess restricted to VOICE_LANGUAGES (never falls back to
    Whisper's full ~99-language auto-detect, which is what previously let
    a low-confidence guess drift to e.g. Russian/Portuguese for this app's
    Urdu/Arabic/English-only users)."""
    try:
        import whisper
        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        candidates = {lang: probs.get(lang, 0.0) for lang in VOICE_LANGUAGES}
        if not candidates:
            return None
        return max(candidates, key=candidates.get)
    except Exception as exc:
        print(f"[whisper] language detection failed: {exc}")
        return VOICE_LANGUAGES[0] if VOICE_LANGUAGES else None


def _boost_quiet_audio(pcm: np.ndarray) -> np.ndarray:
    """Many mics (laptop/phone) record at a low volume, which makes Whisper
    think there's no speech at all. Normalize toward full scale so quiet
    recordings still transcribe correctly, without over-amplifying noise
    (too aggressive a boost turns a near-silent clip's background hiss into
    something loud enough that Whisper tries to "transcribe" it, which is
    what produced repeated-word hallucinations like "tipi, tipi, tipi…")."""
    if pcm.size == 0:
        return pcm
    peak = float(np.abs(pcm).max())
    if peak < 150:  # truly silent/near-empty buffer — nothing useful to boost
        return pcm
    gain = min(30000.0 / peak, 6.0)
    if gain <= 1.05:
        return pcm
    boosted = np.clip(pcm.astype(np.float64) * gain, -32768, 32767)
    return boosted.astype(np.int16)


def _is_repetition_hallucination(text: str) -> bool:
    """Whisper occasionally gets stuck decoding a short phrase over and over
    on noisy/unclear audio (e.g. 'tipi, tipi, tipi, …' x50) instead of
    failing cleanly. Detect a short phrase repeating many times in a row and
    treat it as noise rather than showing it to the user as a real answer."""
    words = re.findall(r"[^\s,]+", text.lower())
    if len(words) < 10:
        return False
    for n in (1, 2, 3):
        limit = len(words) - n * 4
        i = 0
        while i < limit:
            phrase = words[i:i + n]
            repeats = 1
            j = i + n
            while words[j:j + n] == phrase:
                repeats += 1
                j += n
            if repeats >= 5:
                return True
            i += 1
    return False


def transcribe_audio(wav_bytes):
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as file:
            file.write(wav_bytes)
            path = file.name

        model = get_whisper()
        language_hint = _detect_voice_language(model, path)

        result = model.transcribe(
            path, task="transcribe", fp16=False, temperature=0,
            condition_on_previous_text=False,
            **({"language": language_hint} if language_hint else {}),
        )
        text = normalize_text(result.get("text", ""))
        if not text:
            return ""
        if _is_repetition_hallucination(text):
            print(f"[whisper] discarded repetition-hallucination output: {text[:80]}…")
            return ""

        # Whisper already drops low-confidence segments internally while
        # decoding. Re-filtering the *output* with the same thresholds
        # double-penalizes short/quiet-but-real questions (e.g. "dress
        # code?") and is what caused false "no speech detected" results.
        # Only treat it as noise if EVERY segment is both extremely
        # low-confidence AND clearly repetitive/garbled — a much rarer,
        # safer bar than before.
        segments = result.get("segments") or []
        if segments:
            def segment_is_bad(seg):
                silence_like = seg.get("no_speech_prob", 0.0) > 0.85
                garbled = seg.get("compression_ratio", 0.0) > 2.8
                return silence_like and garbled

            if all(segment_is_bad(seg) for seg in segments):
                return ""
        return text
    except Exception as exc:
        print(f"[whisper] transcription failed: {exc}")
        return ""
    finally:
        if path and os.path.exists(path):
            os.remove(path)


def text_to_speech(text, language):
    import edge_tts

    voices = {
        "English": "en-US-AriaNeural",
        "Urdu": "ur-PK-UzmaNeural",
        "Arabic": "ar-SA-ZariyahNeural",
        "Roman Urdu": "ur-PK-UzmaNeural",
    }
    voice = voices.get(language, "en-US-AriaNeural")
    text = str(text).strip()
    if not text:
        return None

    filename = "ai_pdf_tts_" + hashlib.sha1(text.encode("utf-8")).hexdigest() + ".mp3"
    path = os.path.join(tempfile.gettempdir(), filename)

    try:
        async def generate():
            communicate = edge_tts.Communicate(text=text, voice=voice, rate="+0%", volume="+0%")
            await communicate.save(path)

        asyncio.run(generate())

        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return None
        return path
    except Exception as exc:
        print(f"[tts] read aloud failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Chainlit settings + hidden action bridge used by the custom workspace UI
# ---------------------------------------------------------------------------

async def build_settings():
    """Keep Chainlit's native settings available as a fallback."""
    return await cl.ChatSettings([
        cl.input_widget.Select(
            id="answer_mode",
            label="PDF Mode",
            values=["📄 PDF Only", "🌐 PDF + Internet"],
            initial_index=0,
        ),
        cl.input_widget.Slider(
            id="max_sources",
            label="Max Sources",
            initial=FINAL_K,
            min=1,
            max=6,
            step=1,
        ),
        cl.input_widget.Switch(
            id="conversation_memory",
            label="Conversation Memory",
            initial=True,
        ),
    ]).send()


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("answer_mode", settings.get("answer_mode", "📄 PDF Only"))
    cl.user_session.set("max_sources", int(settings.get("max_sources", FINAL_K)))
    cl.user_session.set("conversation_memory_enabled", bool(settings.get("conversation_memory", True)))


async def send_settings_state():
    mode = cl.user_session.get("answer_mode", "📄 PDF Only")
    sources = int(cl.user_session.get("max_sources", FINAL_K))
    memory = bool(cl.user_session.get("conversation_memory_enabled", True))
    await cl.Message(
        content=f"AI_SETTINGS_STATE | {mode} | {sources} | {str(memory).lower()}",
        author="System",
    ).send()


async def send_action_bridge():
    """Render invisible native Chainlit actions for the custom sidebar.

    The browser consumes this message immediately, so users never see the
    bridge controls in the conversation. The callbacks remain server-side and
    preserve Chainlit's reliable websocket action mechanism.
    """
    actions = [
        cl.Action(name="new_chat", payload={}, label="New Chat"),
        cl.Action(name="open_settings", payload={}, label="Controls"),
        cl.Action(name="clear_chat", payload={}, label="Clear History"),
        cl.Action(name="remove_all_pdfs", payload={}, label="Clear PDFs"),
        cl.Action(name="set_pdf_only", payload={}, label="PDF Only"),
        cl.Action(name="set_pdf_internet", payload={}, label="PDF + Internet"),
        cl.Action(name="memory_on", payload={}, label="Memory On"),
        cl.Action(name="memory_off", payload={}, label="Memory Off"),
    ]
    for value in range(1, 7):
        actions.append(cl.Action(name=f"set_sources_{value}", payload={}, label=f"Sources {value}"))
    await cl.Message(content="AI_ACTION_BRIDGE", author="System", actions=actions).send()


@cl.action_callback("set_pdf_only")
async def on_set_pdf_only(action: cl.Action):
    cl.user_session.set("answer_mode", "📄 PDF Only")
    await send_settings_state()


@cl.action_callback("set_pdf_internet")
async def on_set_pdf_internet(action: cl.Action):
    cl.user_session.set("answer_mode", "🌐 PDF + Internet")
    await send_settings_state()


@cl.action_callback("memory_on")
async def on_memory_on(action: cl.Action):
    cl.user_session.set("conversation_memory_enabled", True)
    await send_settings_state()


@cl.action_callback("memory_off")
async def on_memory_off(action: cl.Action):
    cl.user_session.set("conversation_memory_enabled", False)
    await send_settings_state()


@cl.action_callback("set_sources_1")
async def on_set_sources_1(action: cl.Action):
    cl.user_session.set("max_sources", 1)
    await send_settings_state()


@cl.action_callback("set_sources_2")
async def on_set_sources_2(action: cl.Action):
    cl.user_session.set("max_sources", 2)
    await send_settings_state()


@cl.action_callback("set_sources_3")
async def on_set_sources_3(action: cl.Action):
    cl.user_session.set("max_sources", 3)
    await send_settings_state()


@cl.action_callback("set_sources_4")
async def on_set_sources_4(action: cl.Action):
    cl.user_session.set("max_sources", 4)
    await send_settings_state()


@cl.action_callback("set_sources_5")
async def on_set_sources_5(action: cl.Action):
    cl.user_session.set("max_sources", 5)
    await send_settings_state()


@cl.action_callback("set_sources_6")
async def on_set_sources_6(action: cl.Action):
    cl.user_session.set("max_sources", 6)
    await send_settings_state()


# ---------------------------------------------------------------------------
# Sidebar data bridge: uploaded PDFs are rendered only in the custom sidebar
# ---------------------------------------------------------------------------

async def show_pdf_panel():
    pdf_info = cl.user_session.get("pdf_info") or []
    if not pdf_info:
        await cl.Message(content="AI_PDF_INDEX_STATE | 0 files | 0 pages | 0 chunks", author="System").send()
        return
    total_pages = sum(p["pages"] for p in pdf_info)
    total_chunks = sum(p["chunks"] for p in pdf_info)
    lines = [f"AI_PDF_INDEX_STATE | {len(pdf_info)} files | {total_pages} pages | {total_chunks} chunks"]
    for pdf in pdf_info:
        name = html.escape(str(pdf.get("name") or "PDF"))
        pages = int(pdf.get("pages") or 0)
        chunks = int(pdf.get("chunks") or 0)
        lines.append(f"AI_PDF_INDEX_ITEM | {name} | {pages} pages | {chunks} chunks")
    await cl.Message(content="\n".join(lines), author="System").send()


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@cl.action_callback("clear_chat")
async def on_clear_chat(action: cl.Action):
    clear_history()
    cl.user_session.set("messages", [])
    cl.user_session.set("memory", ConversationMemory())
    await cl.Message(content="AI_UI_CLEAR_HISTORY", author="System").send()


@cl.action_callback("remove_all_pdfs")
async def on_remove_all_pdfs(action: cl.Action):
    remove_all_pdfs()
    for key in ("vector_db", "source_bm25", "chunks", "pdf_info", "upload_signature"):
        cl.user_session.set(key, None)
    await cl.Message(content="AI_UI_CLEAR_PDFS", author="System").send()


@cl.action_callback("open_settings")
async def on_open_settings(action: cl.Action):
    await build_settings()


@cl.action_callback("new_chat")
async def on_new_chat(action: cl.Action):
    clear_history()
    cl.user_session.set("messages", [])
    cl.user_session.set("memory", ConversationMemory())
    await cl.Message(content="AI_UI_NEW_CHAT", author="System").send()


@cl.action_callback("read_aloud")
async def on_read_aloud(action: cl.Action):
    text = action.payload.get("text", "")
    language = action.payload.get("language", "English")
    audio_path = await cl.make_async(text_to_speech)(text, language)
    if not audio_path:
        await cl.Message(content="❌ Could not generate audio for this answer.", author="System").send()
        return
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    try:
        os.remove(audio_path)
    except OSError:
        pass
    await cl.Message(
        content="🔊 Read aloud",
        author="Assistant",
        elements=[cl.Audio(name="answer.mp3", content=audio_bytes, mime="audio/mp3")],
    ).send()


# ---------------------------------------------------------------------------
# PDF ingestion (called from on_message whenever files are attached)
# ---------------------------------------------------------------------------

async def handle_pdf_uploads(elements):
    pdf_elements = [
        e for e in elements
        if getattr(e, "mime", "") == "application/pdf" or str(getattr(e, "name", "")).lower().endswith(".pdf")
    ]
    if not pdf_elements:
        return

    uploads = [_Upload(e.name, e.path) for e in pdf_elements if getattr(e, "path", None)]
    if not uploads:
        return

    signature = file_signature(uploads)

    async with cl.Step(name="📚 Processing PDFs and building the search index…", type="tool") as step:
        log_lines = []

        def log(msg):
            log_lines.append(msg)

        try:
            db, source_bm25, chunks, info = await cl.make_async(process_pdfs)(uploads, log)
        except Exception as exc:
            log_lines.append(f"❌ Fatal indexing error: {type(exc).__name__}: {exc}")
            db, source_bm25, chunks, info = None, None, None, []
        step.output = "\n".join(log_lines) or "Done."

    if db is not None:
        cl.user_session.set("vector_db", db)
        cl.user_session.set("source_bm25", source_bm25)
        cl.user_session.set("chunks", chunks)
        cl.user_session.set("pdf_info", info)
        cl.user_session.set("upload_signature", signature)
        await show_pdf_panel()
    else:
        await cl.Message(content="❌ Could not build a search index from the uploaded PDF(s).", author="System").send()


# ---------------------------------------------------------------------------
# Query handling: routes each question to normal RAG, table analysis, or
# chart analysis, then renders the grounded (or refused) answer + sources.
# ---------------------------------------------------------------------------

def _current_pdf_paths(pdf_info):
    return [p["path"] for p in (pdf_info or []) if p.get("path") and os.path.exists(p["path"])]


async def route_and_answer(query, language, history, answer_mode):
    """Returns (answer, sources)."""
    pdf_info = cl.user_session.get("pdf_info") or []
    pdf_paths = _current_pdf_paths(pdf_info)

    # --- 1) Table-shaped questions -----------------------------------------
    if tca.looks_like_table_question(query) and not tca.looks_like_chart_question(query) and pdf_paths:
        async with cl.Step(name="📊 Analyzing table…", type="tool") as step:
            tables = []
            for path in pdf_paths:
                found = await cl.make_async(tca.extract_tables)(path)
                for item in found:
                    item["pdf_path"] = path
                tables.extend(found)

            if tables:
                # Pick the tables that actually match this question (not just
                # "the first 3 tables found") so both the answer and the
                # sources reflect what was really used.
                top_tables = tca.relevant_tables(query, tables, max_tables=3) or tables[:3]

                # 1) Try a deterministic aggregate answer (average/sum/max/...).
                answer = await cl.make_async(tca.answer_table_query)(query, tables, language)
                aggregate_answered = bool(answer) and not _is_table_refusal(answer, language)

                if aggregate_answered:
                    # Humanize the raw deterministic phrasing
                    # (e.g. "Maximum (Salary): 30 — page 2.") into a normal
                    # sentence, without touching genuine LLM-written answers.
                    answer = _humanize_table_answer(answer)
                    # answer_table_query() picks its own best-matching table
                    # internally (e.g. scored by field name / numeric density
                    # for "average salary"), which can differ from the
                    # generic top_tables computed above — cite the table it
                    # ACTUALLY used (parsed from the "page N" it already
                    # embeds in the answer) instead of a possibly-different
                    # one, so the source always matches the real answer.
                    page_match = re.search(r"(?:page|صفحہ|الصفحة)\s*(\d+)", answer)
                    if page_match:
                        cited_page = int(page_match.group(1))
                        matching = [
                            t for t in tables
                            if (t.get("page_range") or (t.get("page"), t.get("page")))[0] <= cited_page
                            <= (t.get("page_range") or (t.get("page"), t.get("page")))[1]
                        ]
                        if matching:
                            top_tables = matching[:1]
                else:
                    # 2) No aggregate matched (e.g. "show me the table" /
                    # "summary table"). Build the summary + the real table
                    # ourselves using the table's ACTUAL column headers, so we
                    # never show placeholder labels like "col_0" or numbers
                    # invented by the model.
                    primary = top_tables[0]
                    df = primary.get("dataframe")
                    summary = _deterministic_table_summary(df, language)
                    table_md = _df_to_markdown(df, max_rows=20)
                    parts = [p for p in (summary, table_md) if p]
                    answer = "\n\n".join(parts)

                    if not answer:
                        # Last resort only: nothing deterministic could be
                        # produced, ask the model for a grounded narrative.
                        answer = await cl.make_async(_answer_table_with_llm)(query, tables, language)

                if not answer or not answer.strip():
                    answer = "I found table data, but could not produce an answer for that question."

                step.output = f"Found {len(tables)} table(s) across {len(pdf_paths)} PDF(s)."
                table_sources = []
                seen_table_sources = set()
                for item in top_tables:
                    page_range = item.get("page_range") or (item.get("page"), item.get("page"))
                    page_label = (
                        f"{page_range[0]}–{page_range[1]}" if page_range[0] != page_range[1]
                        else item.get("page", "table")
                    )
                    entry = {
                        "file": os.path.basename(item.get("pdf_path") or "PDF"),
                        "page": page_label,
                    }
                    key = (entry["file"], str(entry["page"]))
                    if key not in seen_table_sources:
                        seen_table_sources.add(key)
                        table_sources.append(entry)
                return answer, table_sources

            step.output = "No machine-readable table was extracted. Falling back to PDF text search."

    # --- 2) Chart-shaped questions -------------------------------------------
    if tca.looks_like_chart_question(query) and pdf_paths:
        async with cl.Step(name="📈 Analyzing chart…", type="tool") as step:
            candidates = []  # [(path, pages), ...] for every PDF with likely chart pages
            for path in pdf_paths:
                pages = await cl.make_async(tca.find_chart_pages)(path)
                if pages:
                    candidates.append((path, pages))

            if candidates:
                if tca.looks_like_all_charts_request(query):
                    sections = []
                    sources = []
                    for path, pages in candidates:
                        result = await cl.make_async(tca.analyze_all_charts)(
                            path, pages, language, gemini_vision_call
                        )
                        if result and result not in tca.REFUSAL_NO_TABLE.values():
                            sections.append(result)
                            sources.extend({"file": os.path.basename(path), "page": p} for p in pages)
                    step.output = f"Analyzed charts across {len(candidates)} PDF(s)."
                    answer = "\n\n---\n\n".join(sections) if sections else (
                        await cl.make_async(tca.analyze_all_charts)(*candidates[0], language, gemini_vision_call)
                    )
                    return answer, sources

                # Not all charts are necessarily in the first PDF that merely
                # *has* chart-like pages — try each candidate PDF in turn
                # until one actually answers the question, instead of giving
                # up after the first file (which is what caused "pie chart"
                # to fail when the pie chart lived in a different upload
                # than whichever PDF happened to be checked first).
                for path, pages in candidates:
                    answer_text, used_page = await cl.make_async(tca.answer_chart_query)(
                        query, path, pages, language, gemini_vision_call
                    )
                    if used_page is not None:
                        step.output = f"Inspected {len(pages)} page(s) with charts/figures in {os.path.basename(path)}."
                        return answer_text, [{"file": os.path.basename(path), "page": used_page}]
                # Genuinely nothing found in any candidate PDF — don't cite a
                # page as a "source" when nothing was actually read from it;
                # that was misleading (e.g. citing "Page 2" for a chart that
                # doesn't exist there).
                step.output = f"Checked {len(candidates)} PDF(s) but none had the requested chart."
                return answer_text, []
            step.output = "No chart/figure pages found — falling back to normal PDF search."

    # --- 3) Normal grounded PDF RAG -----------------------------------------
    vector_db = cl.user_session.get("vector_db")
    source_bm25 = cl.user_session.get("source_bm25")
    chunks = cl.user_session.get("chunks")

    async with cl.Step(name="🔎 Searching your PDFs…", type="tool") as step:
        try:
            rewritten, results = await cl.make_async(retrieve_documents)(
                query, vector_db, source_bm25, chunks, history,
                retrieval_k=RETRIEVAL_K, final_k=cl.user_session.get("max_sources", FINAL_K),
            )
        except Exception as exc:
            step.output = f"❌ PDF search failed: {exc}"
            return refusal(language), []

        if not results:
            step.output = "No relevant PDF passages found."
            answer, sources = refusal(language), []
            if answer_mode == "🌐 PDF + Internet":
                answer, sources = await internet_fallback(query, language, step)
            return answer, sources

        step.output = "🧠 Reading relevant PDF context and preparing the answer…"

        context = compress_context(rewritten, results, max_chars=5000)
        answer = exact_field_answer(query, results, language)

        if answer:
            results = results[:1]
        else:
            try:
                llm = load_llm()
                answer = await cl.make_async(generate_grounded_answer)(
                    llm, query, context, format_history(history), language
                )
            except LLMGenerationError as exc:
                step.output = f"❌ AI model error: {exc}"
                raise
            except Exception as exc:
                step.output = f"❌ Unexpected AI model error: {type(exc).__name__}: {exc}"
                raise

    validation = validate_answer(query, context, answer)

    if not answer or is_refusal(answer) or validation == "NOT_SUPPORTED":
        answer, sources = refusal(language), []
        if answer_mode == "🌐 PDF + Internet":
            async with cl.Step(name="🌐 Searching Internet sources…", type="tool") as step:
                answer, sources = await internet_fallback(query, language, step)
        return answer, sources

    if validation == "PARTIALLY_SUPPORTED":
        partial = {
            "English": "I can only partially answer this from the PDF.\n\n",
            "Urdu": "میں PDF میں موجود معلومات کی بنیاد پر صرف جزوی جواب دے سکتا ہوں۔\n\n",
            "Roman Urdu": "Main PDF mein mojood maloomat ki bunyaad par sirf kuch hissa bata sakta hoon.\n\n",
            "Arabic": "يمكنني الإجابة جزئيًا فقط بناءً على المعلومات الموجودة في ملف PDF.\n\n",
        }[language]
        answer = partial + answer

    if not answer_is_safe(answer, language):
        return refusal(language), []

    return answer, source_info(_only_used_sources(answer, results))


async def handle_query(query):
    query = normalize_text(query)
    if not query:
        return

    language = detect_language(query)
    messages = cl.user_session.get("messages") or []
    memory: ConversationMemory = cl.user_session.get("memory") or ConversationMemory()

    messages.append({"role": "user", "content": query, "sources": []})
    save_message("user", query, [])

    if not cl.user_session.get("vector_db"):
        answer = refusal(language)
        await cl.Message(content=answer, author="Assistant").send()
        messages.append({"role": "assistant", "content": answer, "sources": []})
        save_message("assistant", answer, [])
        memory.add_message(query, answer)
        cl.user_session.set("messages", messages)
        cl.user_session.set("memory", memory)
        return

    history = memory.get_history() if cl.user_session.get("conversation_memory_enabled", True) else []
    answer_mode = cl.user_session.get("answer_mode", "📄 PDF Only")
    start = time.time()

    answer, sources = await route_and_answer(query, language, history, answer_mode)

    elapsed = time.time() - start
    # Never render a message whose visible body consists only of citations.
    # A defensive fallback makes source-only responses impossible when an
    # upstream table/chart/parser unexpectedly returns an empty string.
    if not answer or not str(answer).strip():
        answer = refusal(language)
    content = str(answer).strip() + sources_markdown(sources)
    content += f"\n\n*⏱️ {elapsed:.1f}s*"

    actions = [
        cl.Action(
            name="read_aloud",
            payload={"text": answer, "language": language},
            label="🔊 Read Aloud",
        )
    ]
    await cl.Message(content=content, author="Assistant", actions=actions).send()

    messages.append({"role": "assistant", "content": answer, "sources": sources})
    save_message("assistant", answer, sources)
    memory.add_message(query, answer)
    cl.user_session.set("messages", messages)
    cl.user_session.set("memory", memory)


# ---------------------------------------------------------------------------
# Chainlit lifecycle hooks
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    init_db()

    messages = load_messages()
    cl.user_session.set("messages", messages)

    memory = ConversationMemory()
    pairs = zip(messages[0::2], messages[1::2])
    for user_msg, assistant_msg in pairs:
        if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
            memory.add_message(user_msg["content"], assistant_msg["content"])
    cl.user_session.set("memory", memory)

    cl.user_session.set("answer_mode", "📄 PDF Only")
    cl.user_session.set("max_sources", FINAL_K)
    cl.user_session.set("conversation_memory_enabled", True)
    cl.user_session.set("upload_signature", None)

    db, source_bm25, chunks, info = await cl.make_async(load_saved_index)()
    if db is not None:
        cl.user_session.set("vector_db", db)
        cl.user_session.set("source_bm25", source_bm25)
        cl.user_session.set("chunks", chunks)
        cl.user_session.set("pdf_info", info)

    welcome = (
        "**Welcome to AI PDF Chatbot**\n\n"
        "Upload one or more PDFs and ask questions in your preferred language. "
        "Answers stay grounded in your documents with page citations, hybrid retrieval, "
        "voice input and read-aloud.\n\n"
        "📎 **Upload a PDF** to begin, then ask your question below."
    )
    await cl.Message(content=welcome, author="Assistant").send()

    await send_action_bridge()
    await send_settings_state()
    await show_pdf_panel()

    # Warm up the Whisper model in the background so the *first* voice
    # recording doesn't sit at "Transcribing…" while the model loads.
    async def _warm_up_whisper():
        try:
            await cl.make_async(get_whisper)()
        except Exception as exc:
            print(f"[voice] whisper warm-up failed: {exc}")
    asyncio.create_task(_warm_up_whisper())

    # Replay prior conversation so refreshing the page doesn't lose history.
    for message in messages:
        author = "Assistant" if message["role"] == "assistant" else "You"
        await cl.Message(content=message["content"], author=author).send()


def _friendly_llm_error(exc):
    """Turn a raw Gemini exception into a clean, non-technical message.
    Falls back to the raw (but NOT re-HTML-escaped — Chainlit's markdown
    renderer already escapes safely) exception text for anything else."""
    text = str(exc)
    if "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
        return (
            "**Gemini API usage limit reached (HTTP 429).**\n\n"
            "Your Gemini API key has hit its current quota/rate limit — this is "
            "an account/billing limit on the API key, not a bug in the app. "
            "Options:\n"
            "- Wait a bit for the quota to reset (free-tier quotas usually reset daily/per-minute).\n"
            "- Check your plan & billing at https://ai.google.dev/gemini-api/docs/rate-limits\n"
            "- Use a different `GEMINI_API_KEY` with a higher quota, or switch "
            "`AI_PDF_GEMINI_MODEL` to a lower-cost model in `.env`."
        )
    return f"`{text}`"


@cl.on_message
async def on_message(message: cl.Message):
    try:
        if message.elements:
            await handle_pdf_uploads(message.elements)

        text = normalize_text(message.content)
        if text:
            await handle_query(text)
    except LLMGenerationError as exc:
        await cl.Message(
            content=(
                "❌ **AI model error**\n\n"
                f"{_friendly_llm_error(exc)}\n\n"
                "If this isn't a quota/rate-limit message, check `GEMINI_API_KEY` "
                "(or `GOOGLE_API_KEY`) and `AI_PDF_GEMINI_MODEL` in `.env`, then restart the app."
            ),
            author="System",
        ).send()
    except Exception as exc:
        print(f"[app] unhandled message error: {type(exc).__name__}: {exc}")
        await cl.Message(
            content=(
                "❌ **Something went wrong while processing your request.**\n\n"
                f"`{type(exc).__name__}: {exc}`"
            ),
            author="System",
        ).send()


# --- Voice input: mic button in the Chainlit composer ------------------------

@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("audio_chunks", [])
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")
    if audio_chunks is not None:
        audio_chunks.append(np.frombuffer(chunk.data, dtype=np.int16))


@cl.on_audio_end
async def on_audio_end():
    audio_chunks = cl.user_session.get("audio_chunks") or []
    cl.user_session.set("audio_chunks", [])
    if not audio_chunks:
        return

    try:
        pcm = np.concatenate(audio_chunks)

        duration_seconds = len(pcm) / float(WHISPER_SAMPLE_RATE)
        if duration_seconds < 1.0:
            await cl.Message(
                content="🎙️ That recording was too short to catch anything — please hold the mic button and speak for a bit longer.",
                author="System",
            ).send()
            return

        pcm = _boost_quiet_audio(pcm)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(WHISPER_SAMPLE_RATE)
            wav_file.writeframes(pcm.tobytes())
        wav_bytes = wav_buffer.getvalue()

        # TEMPORARY diagnostic aid: set AI_PDF_VOICE_DEBUG=true in .env to
        # attach the exact recorded WAV as a downloadable file, so it can be
        # inspected directly instead of guessing blind. Safe to remove/leave
        # off — default is off and it changes no transcription behavior.
        if os.getenv("AI_PDF_VOICE_DEBUG", "false").strip().lower() == "true":
            await cl.Message(
                content=f"🐞 Debug: raw recording — {duration_seconds:.1f}s, {len(pcm)} samples @ {WHISPER_SAMPLE_RATE}Hz.",
                author="System",
                elements=[cl.File(name="debug_recording.wav", content=wav_bytes, display="inline")],
            ).send()

        text = ""
        async with cl.Step(name="🎙️ Transcribing your recording…", type="tool") as step:
            try:
                # The first recording of the session also has to load the
                # Whisper model, which can take a while — cap it so the UI
                # never spins forever instead of failing loudly.
                text = await asyncio.wait_for(cl.make_async(transcribe_audio)(wav_bytes), timeout=120)
                step.output = f"Heard: “{text}”" if text else "No clear speech detected in that recording."
            except asyncio.TimeoutError:
                step.output = "Timed out while transcribing (the voice model may still be loading)."
                await cl.Message(
                    content=(
                        "🎙️ Transcription timed out — the voice model may still be loading on the server "
                        "for the first time. Please wait a few seconds and try recording again."
                    ),
                    author="System",
                ).send()
                return

        if not text:
            await cl.Message(
                content=(
                    "🎙️ I couldn't make out any speech in that recording. "
                    "Please check your microphone permissions/volume and try again, "
                    "speaking clearly and a little closer to the mic."
                ),
                author="System",
            ).send()
            return

        # Put the transcribed speech into the message box itself (like a
        # normal voice-to-text field) so the user can see exactly what was
        # heard, edit it (type more, delete, select-all, replace — the box
        # behaves like normal typing) and press Send themselves.
        await cl.Message(content=f"AI_VOICE_TRANSCRIPT|{text}", author="System").send()
    except Exception as exc:
        # Whatever goes wrong, always resolve the pending recording UI
        # instead of leaving "Transcribing your recording…" stuck forever.
        print(f"[voice] on_audio_end failed: {type(exc).__name__}: {exc}")
        await cl.Message(
            content="🎙️ Voice input failed unexpectedly. Please try again, or type your question instead.",
            author="System",
        ).send()
