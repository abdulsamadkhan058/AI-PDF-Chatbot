"""Production-safe hybrid lexical + semantic retrieval helpers.

Combines:
- FAISS semantic retrieval
- BM25 lexical retrieval
- Safe deduplication
- Multilingual text support
- Defensive error handling

No LLM call is made here.
"""

from copy import copy

from langchain_community.retrievers import BM25Retriever


DEFAULT_K = 8
MAX_K = 50


def _safe_int(value, default=DEFAULT_K):
    """Convert a value to a safe positive integer."""
    try:
        value = int(value)
        return max(1, min(value, MAX_K))
    except (TypeError, ValueError):
        return default
    except Exception:
        return default


def _doc_key(doc):
    """Create a stable key for document deduplication."""

    try:
        metadata = getattr(
            doc,
            "metadata",
            {},
        ) or {}

        source = str(
            metadata.get(
                "source",
                "",
            )
            or ""
        ).replace(
            "\\",
            "/",
        )

        try:
            page = int(
                metadata.get(
                    "page",
                    0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            page = 0

        content = str(
            getattr(
                doc,
                "page_content",
                "",
            )
            or ""
        ).strip()

        return (
            source,
            page,
            content,
        )

    except Exception:
        # Last-resort key.
        try:
            return (
                "",
                0,
                str(doc or ""),
            )
        except Exception:
            return (
                "",
                0,
                "",
            )


def deduplicate_documents(documents):
    """Safely remove duplicate LangChain documents."""

    if not documents:
        return []

    seen = set()
    output = []

    try:
        for doc in documents:

            if doc is None:
                continue

            try:
                key = _doc_key(doc)

                if key in seen:
                    continue

                seen.add(key)
                output.append(doc)

            except Exception:
                # Ignore only the malformed document.
                continue

    except (
        TypeError,
        AttributeError,
    ):
        return output

    except Exception:
        return output

    return output


def create_bm25_retriever(
    chunks,
    k=DEFAULT_K,
):
    """Create a BM25 retriever from PDF chunks.

    Raises:
        ValueError:
            If chunks are empty.
        RuntimeError:
            If BM25 creation fails.
    """

    if not chunks:
        raise ValueError(
            "Cannot create BM25 retriever from empty chunks."
        )

    k = _safe_int(k)

    try:
        retriever = (
            BM25Retriever.from_documents(
                chunks
            )
        )

        retriever.k = k

        return retriever

    except Exception as exc:
        raise RuntimeError(
            "Failed to create BM25 retriever."
        ) from exc


def _semantic_search(
    query,
    vector_db,
    k,
):
    """Run FAISS semantic search safely."""

    if vector_db is None:
        return []

    try:
        results = (
            vector_db.similarity_search(
                query,
                k=k,
            )
        )

        return results or []

    except Exception:
        return []


def _lexical_search(
    query,
    bm25_retriever,
    k,
):
    """Run BM25 lexical search safely."""

    if bm25_retriever is None:
        return []

    try:
        
        local = copy(
            bm25_retriever
        )

        local.k = k

        results = local.invoke(
            query
        )

        return results or []

    except Exception:
        return []


def hybrid_search(
    query,
    vector_db=None,
    bm25_retriever=None,
    k=DEFAULT_K,
):
    """Combine FAISS semantic and BM25 lexical retrieval.

    The function is intentionally failure-tolerant:

    - FAISS failure does not stop BM25.
    - BM25 failure does not stop FAISS.
    - Empty query returns [].
    - Missing retrievers return [].
    - Duplicate chunks are removed.
    """

    try:
        query = str(
            query or ""
        ).strip()

    except Exception:
        return []

    if not query:
        return []

    if (
        vector_db is None
        and bm25_retriever is None
    ):
        return []

    k = _safe_int(k)

    semantic_results = _semantic_search(
        query=query,
        vector_db=vector_db,
        k=k,
    )

    lexical_results = _lexical_search(
        query=query,
        bm25_retriever=bm25_retriever,
        k=k,
    )

    return deduplicate_documents(
        [
            *semantic_results,
            *lexical_results,
        ]
    )