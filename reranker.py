"""Production-safe deterministic document reranker."""

import re

_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+"
    r"|"
    r"[\u0600-\u06FF"
    r"\u0750-\u077F"
    r"\u08A0-\u08FF"
    r"\uFB50-\uFDFF"
    r"\uFE70-\uFEFF]+"
)

_STOP_WORDS = {
    # English
    "the", "and", "for", "what", "when", "where",
    "which", "who", "why", "how", "does", "is",
    "are", "was", "were", "this", "that", "these",
    "those", "with", "from", "about", "tell", "give",
    "please", "can", "you", "your", "their", "they",
    "them", "not", "a", "an", "to", "of", "in",
    "on", "as", "at", "or", "it",

    # Urdu
    "ہے", "ہیں", "کیا", "کا", "کی", "کے", "سے",
    "کو", "اور", "یہ", "وہ", "اس", "ان", "میں",
    "پر", "نے", "تھا", "تھی", "تھے",

    # Arabic
    "هل", "ما", "ماذا", "كيف", "من", "في", "على",
    "عن", "هو", "هي", "هذا", "هذه", "ذلك", "تلك",
}


def _tokens(text):
    """Extract normalized multilingual tokens safely."""

    try:
        tokens = _TOKEN_PATTERN.findall(
            str(text or "").lower()
        )

        return {
            token
            for token in tokens
            if len(token) >= 2
            and token not in _STOP_WORDS
        }

    except Exception:
        return set()


def _safe_document_text(doc):
    """Safely get document text."""

    try:
        return str(
            getattr(
                doc,
                "page_content",
                "",
            )
            or ""
        ).strip()

    except Exception:
        return ""


def _source(doc):
    """Safely extract document source."""

    try:
        metadata = getattr(
            doc,
            "metadata",
            {},
        ) or {}

        source = str(
            metadata.get(
                "source",
                "Unknown",
            )
            or "Unknown"
        )

        return source.replace(
            "\\",
            "/",
        ).split("/")[-1]

    except Exception:
        return "Unknown"


def _overlap_score(query_tokens, doc_tokens):
    """Calculate lexical overlap."""

    if not query_tokens or not doc_tokens:
        return 0.0

    overlap = len(
        query_tokens & doc_tokens
    )

    return overlap / max(
        1,
        len(query_tokens),
    )


def _phrase_score(query, text):
    """Give a small boost for exact multi-word matches."""

    try:
        normalized_query = (
            str(query or "")
            .strip()
            .lower()
        )

        normalized_text = (
            str(text or "")
            .strip()
            .lower()
        )

        if (
            not normalized_query
            or not normalized_text
        ):
            return 0.0

        query_tokens = _tokens(
            normalized_query
        )

        if len(query_tokens) < 2:
            return 0.0

        compact_query = " ".join(
            query_tokens
        )

        compact_text = " ".join(
            _tokens(normalized_text)
        )

        if compact_query and compact_query in compact_text:
            return 1.0

    except Exception:
        pass

    return 0.0

def _score_document(query, doc):
   
    text = _safe_document_text(
        doc
    )

    if not text:
        return 0.0

    query_tokens = _tokens(
        query
    )

    doc_tokens = _tokens(
        text
    )

    overlap = _overlap_score(
        query_tokens,
        doc_tokens,
    )

    phrase = _phrase_score(
        query,
        text,
    )

    # Lexical relevance.
    score = (
        overlap * 10.0
        + phrase * 3.0
    )

    return score


def rerank_documents(
    query,
    documents,
    top_k=6,
):
    
    if not documents:
        return []

    try:
        top_k = max(
            1,
            min(
                int(top_k),
                50,
            ),
        )

    except (
        TypeError,
        ValueError,
    ):
        top_k = 6

    try:
        query = str(
            query or ""
        ).strip()

    except Exception:
        return list(
            documents[:top_k]
        )

    if not query:
        return list(
            documents[:top_k]
        )

    scored = []

    try:
        for index, doc in enumerate(
            documents
        ):
            try:
                score = _score_document(
                    query,
                    doc,
                )

                scored.append(
                    (
                        score,
                        index,
                        doc,
                    )
                )

            except Exception:
                
                scored.append(
                    (
                        0.0,
                        index,
                        doc,
                    )
                )

    except Exception:
        return list(
            documents[:top_k]
        )

    scored.sort(
        key=lambda item: (
            item[0],
            -item[1],
        ),
        reverse=True,
    )

    return [
        item[2]
        for item in scored[:top_k]
    ]