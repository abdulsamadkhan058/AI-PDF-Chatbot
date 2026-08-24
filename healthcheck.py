"""Production sanity checks for AI PDF Chatbot.

This script does not call Gemini or download models. It validates files,
Python version, configuration, and import availability so deployment issues
are caught before starting Chainlit.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "app.py",
    "requirements.txt",
    ".chainlit/config.toml",
    "public/style.css",
    "core/llm.py",
    "core/embeddings.py",
    "retrieval.py",
]

REQUIRED_MODULES = [
    "chainlit",
    "numpy",
    "pypdf",
    "langchain_core",
    "langchain_community",
    "langchain_text_splitters",
    "langchain_huggingface",
    "sentence_transformers",
    "faiss",
    "google.genai",
    "ddgs",
    "pdfplumber",
    "fitz",
]

def main() -> int:
    errors = []
    print("AI PDF Chatbot health check")
    print("=" * 32)
    print(f"Python: {sys.version.split()[0]}")
    if not (sys.version_info >= (3, 10) and sys.version_info < (3, 14)):
        errors.append("Python 3.10–3.13 is required.")

    for rel in REQUIRED_FILES:
        ok = (ROOT / rel).is_file()
        print(f"{'OK' if ok else 'MISS':4} {rel}")
        if not ok:
            errors.append(f"Missing required file: {rel}")

    print("\nDependencies")
    for module in REQUIRED_MODULES:
        ok = importlib.util.find_spec(module) is not None
        print(f"{'OK' if ok else 'MISS':4} {module}")
        if not ok:
            errors.append(f"Missing dependency: {module}")

    env_file = ROOT / ".env"
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key and env_file.exists():
        # Avoid importing dotenv just for a check; presence is enough here.
        text = env_file.read_text(encoding="utf-8", errors="ignore")
        key = next(
            (line.split("=", 1)[1].strip() for line in text.splitlines()
             if (line.startswith("GOOGLE_API_KEY=") or line.startswith("GEMINI_API_KEY=")) and "=" in line),
            "",
        )
    print(f"ENV  GOOGLE_API_KEY/GEMINI_API_KEY {'configured' if key and 'your_gemini' not in key else 'NOT CONFIGURED'}")
    if not key or "your_gemini" in key:
        errors.append("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env before asking questions.")

    print("\nResult")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("READY: dependency/config sanity checks passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
