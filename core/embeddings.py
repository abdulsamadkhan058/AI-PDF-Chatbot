"""Cached multilingual embedding model."""

import os
from functools import lru_cache
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


def _get_env(name, default):
    """Read an environment variable safely."""
    value = os.getenv(name, default)

    if value is None:
        return default

    value = str(value).strip()

    return value or default


@lru_cache(maxsize=1)
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
        print(f"❌ {message}")
        return None