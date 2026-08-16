"""Cached multilingual embedding model."""

import os
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

try:
    import streamlit as st
except ImportError:
    st = None


def _cache(func):
    if st is not None:
        return st.cache_resource(
            show_spinner="🧠 Loading multilingual embedding model…"
        )(func)

    return lru_cache(maxsize=1)(func)


def _get_env(name, default):
    """Read an environment variable safely."""
    value = os.getenv(name, default)

    if value is None:
        return default

    value = str(value).strip()

    return value or default


@_cache
def load_embedding():
    model_name = _get_env(
        "AI_PDF_EMBEDDING_MODEL",
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2",
    )

    device = _get_env(
        "AI_PDF_EMBEDDING_DEVICE",
        "cpu",
    )

    try:
        embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": device,
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

        return embedding

    except Exception as exc:
        message = (
            "Failed to load embedding model "
            f"'{model_name}' on device '{device}': {exc}"
        )

        if st is not None:
            st.error(f"❌ {message}")

        else:
            print(f"❌ {message}")

        return None