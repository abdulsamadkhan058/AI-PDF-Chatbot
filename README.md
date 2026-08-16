# AI PDF Bot — Final

A professional Streamlit PDF chatbot with:

- English, Urdu, Arabic and Roman Urdu routing
- RTL rendering for Urdu/Arabic
- Multilingual embeddings
- Source-aware FAISS + per-document BM25 retrieval
- Deterministic follow-up query rewriting
- Embedding reranking
- Context compression
- Grounded answer validation
- Multi-PDF support
- SQLite chat history
- Native chat-input microphone transcription with Whisper
- Optional PDF + Internet mode
- Responsive mobile-friendly UI
- No text-to-speech

## Windows setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Start Ollama:

```powershell
ollama serve
ollama pull qwen2.5:3b
```

Run:

```powershell
streamlit run streamlit_app.py
```

## Recommended tests

Use **PDF Only** first.

Recommended Tests

Start with PDF Only.

English:
What is the annual leave policy?

Follow-up:
How many days of annual leave are provided?

Urdu:
سالانہ چھٹیوں کی پالیسی کیا ہے؟

Roman Urdu:
Meri annual leave ki policy kya hai?

Arabic:
ما هي سياسة الإجازة السنوية؟

Source Verification:
Check that Sources contain only the document/page used for the final context.

Grounding Test:
Ask a question that is not supported by the uploaded PDF. The assistant should refuse instead of inventing an answer.

## Fresh indexing

When the uploaded PDF set changes, the application rebuilds the FAISS/BM25 index. Existing saved indexes can be removed from the sidebar with **Remove All PDFs**.

## Whisper

Use the microphone inside the chat input and grant browser microphone permission.
Whisper voice transcription may require FFmpeg on Windows. If FFmpeg is already
installed and available in PATH, no extra configuration is needed.

The native chat recorder requires Streamlit 1.52 or newer.

### Mobile microphone requirement

Android Chrome blocks microphone recording on a local-network HTTP address such
as `http://192.168.x.x:8501`; this browser security restriction produces
`Recording failed`. Use `localhost` on the same device for local work, or put
the app behind HTTPS before testing voice from a phone. Allow microphone access
in Chrome after opening the HTTPS address. This is required for any web app,
not a PDF-bot setting.

The default Whisper `small` model prioritizes transcription quality. If the
computer is too slow, set `AI_PDF_WHISPER_MODEL=base` in a local `.env` file.

## Performance

The app uses a short-context, one-pass hybrid retrieval path. Literal fields
such as `Iqama Status` are answered directly from the selected PDF line, so
they do not wait for the local LLM. For faster general answers, use a smaller
Ollama model through `AI_PDF_MODEL` (for example `qwen2.5:1.5b`).
