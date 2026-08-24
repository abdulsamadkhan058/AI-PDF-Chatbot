"""Production multilingual hybrid document retrieval."""

import os
import re

from hybrid_search import (
    create_bm25_retriever,
    deduplicate_documents,
)
from query_rewriter import rewrite_query
from query_transform import transform_query
from reranker import rerank_documents

# Multilingual stop words

_STOP_WORDS = {
    # English
    "the", "and", "for", "what", "when", "where", "which",
    "who", "why", "how", "does", "is", "are", "was", "were",
    "this", "that", "these", "those", "with", "from", "about",
    "tell", "give", "please", "can", "you", "your", "their",
    "they", "them", "not", "a", "an", "to", "of", "in", "on",
    "as", "at", "or", "it",

    # Urdu
    "ہے", "ہیں", "تھا", "تھی", "تھے", "کیا", "کا", "کی", "کے",
    "سے", "کو", "اور", "یہ", "وہ", "اس", "ان", "میں", "پر",
    "نے", "ایک", "کےلیے", "لیے",

    # Arabic
    "هل", "ما", "ماذا", "كيف", "من", "في", "على", "عن",
    "هو", "هي", "هذا", "هذه", "ذلك", "تلك", "و", "من",
    "إلى", "عن", "مع", "هل",
}
# Multilingual hints for multi-document questions

_MULTI_HINTS = {
    # English
    "compare",
    "comparison",
    "both",
    "all",
    "across",
    "different pdfs",
    "different documents",
    "documents",
    "pdfs",
    "between",

    # Roman Urdu
    "dono",
    "donon",
    "sab",
    "mukabla",
    "muqabla",
    "farq",
    "difference",
    "compare karo",
    "dono documents",
    "dono pdf",

    # Urdu
    "دونوں",
    "تمام",
    "موازنہ",
    "مقابلہ",
    "فرق",
    "مختلف",
    "دستاویزات",

    # Arabic
    "كلاهما",
    "مقارنة",
    "الفرق",
    "جميع",
    "كل",
    "مختلف",
    "بين",
    "الوثائق",
    "المستندات",
}
# Tokenization

def _tokens(text):
    """
    Extract multilingual lexical tokens.

    Supports:
    - English
    - numbers
    - Urdu
    - Arabic
    - Arabic-derived Unicode ranges
    """

    try:
        raw = re.findall(
            r"[A-Za-z0-9]+|"
            r"[\u0600-\u06FF"
            r"\u0750-\u077F"
            r"\u08A0-\u08FF"
            r"\uFB50-\uFDFF"
            r"\uFE70-\uFEFF]+",
            str(text or "").lower(),
        )

        return {
            token
            for token in raw
            if len(token) >= 2
            and token not in _STOP_WORDS
        }

    except (
        TypeError,
        AttributeError,
        ValueError,
        re.error,
    ):
        return set()

    except Exception:
        return set()

# Source helpers

def _source(doc):
    """Return a safe PDF source filename."""

    try:
        metadata = getattr(
            doc,
            "metadata",
            {},
        ) or {}

        source = metadata.get(
            "source",
            "Unknown",
        )

        return os.path.basename(
            str(source).replace("\\", "/")
        ) or "Unknown"

    except Exception:
        return "Unknown"


def _source_groups(chunks):
    """Group chunks by PDF source."""

    groups = {}

    for doc in chunks or []:
        try:
            source = _source(doc)

            groups.setdefault(
                source,
                [],
            ).append(doc)

        except Exception:
            continue

    return groups


def build_source_bm25(chunks, k=8):
    """
    Build one BM25 retriever per PDF.

    This prevents a large document collection from allowing
    one PDF to dominate lexical retrieval.
    """

    groups = _source_groups(chunks)

    output = {}

    for source, docs in groups.items():
        try:
            if docs:
                output[source] = create_bm25_retriever(
                    docs,
                    k=k,
                )
        except Exception:
            # One bad PDF should not prevent other PDFs
            # from participating in retrieval.
            continue

    return output

# Lexical evidence

def _lexical_score(query, text):
    """
    Calculate deterministic lexical evidence.

    Returns:
        overlap,
        coverage,
        score
    """

    query_terms = _tokens(query)
    body_terms = _tokens(text)

    if not query_terms or not body_terms:
        return 0, 0.0, 0.0

    overlap = len(
        query_terms & body_terms
    )

    coverage = (
        overlap /
        max(1, len(query_terms))
    )

    # Stronger evidence for more matching terms.
    score = (
        2.0 * coverage
        + 0.40 * min(overlap, 8)
    )

    return (
        overlap,
        coverage,
        score,
    )


def _document_relevance(query, docs):
    """
    Determine whether a PDF has lexical evidence for a query.

    Semantic similarity is useful for candidate discovery,
    but lexical evidence is required before selecting a PDF.
    """

    query_terms = _tokens(query)

    if not query_terms:
        return False, 0, 0.0

    best_overlap = 0
    best_coverage = 0.0

    for doc in docs or []:
        try:
            overlap, coverage, _ = _lexical_score(
                query,
                getattr(
                    doc,
                    "page_content",
                    "",
                ),
            )

            best_overlap = max(
                best_overlap,
                overlap,
            )

            best_coverage = max(
                best_coverage,
                coverage,
            )

        except Exception:
            continue

    if best_overlap <= 0:
        return (
            False,
            best_overlap,
            best_coverage,
        )

    return (
        True,
        best_overlap,
        best_coverage,
    )

# Query helpers

def _is_multi_document_query(query):
    """Detect whether the user explicitly asks about multiple PDFs."""

    try:
        text = str(query or "").lower()

        return any(
            hint in text
            for hint in _MULTI_HINTS
        )

    except Exception:
        return False


def _safe_variants(query):
    """Generate retrieval variants safely."""

    try:
        variants = transform_query(query)

        if not variants:
            return [query]

        # Remove duplicates while preserving order.
        output = []

        seen = set()

        for item in variants:
            value = str(item or "").strip()

            if not value:
                continue

            key = value.lower()

            if key not in seen:
                seen.add(key)
                output.append(value)

        return output or [query]

    except Exception:
        return [query]

# Main retrieval function

def retrieve_documents(
    query,
    vector_db,
    source_bm25,
    chunks,
    history,
    retrieval_k=10,
    final_k=6,
):
    """
    Production multilingual retrieval pipeline.

    Pipeline:

        user query
            ↓
        follow-up rewriting
            ↓
        multilingual query variants
            ↓
        FAISS semantic retrieval
            +
        per-PDF BM25 retrieval
            ↓
        deduplication
            ↓
        PDF evidence scoring
            ↓
        source selection
            ↓
        reranking
            ↓
        final chunks

    Returns:
        (rewritten_query, documents)
    """

    original_query = str(
        query or ""
    ).strip()

    if not original_query:
        return "", []

    # Validate retrieval parameters

    try:
        retrieval_k = max(
            1,
            min(
                int(retrieval_k),
                50,
            ),
        )
    except (
        TypeError,
        ValueError,
    ):
        retrieval_k = 10

    try:
        final_k = max(
            1,
            min(
                int(final_k),
                20,
            ),
        )
    except (
        TypeError,
        ValueError,
    ):
        final_k = 6

    # Validate retrieval resources

    if vector_db is None and not source_bm25:
        return original_query, []

    if not chunks:
        return original_query, []

    # 1. Rewrite follow-up question

    try:
        rewritten = rewrite_query(
            original_query,
            history,
        )

    except Exception:
        rewritten = original_query

    rewritten = str(
        rewritten or original_query
    ).strip()

    if not rewritten:
        rewritten = original_query

    # 2. Create multilingual variants

    variants = _safe_variants(
        rewritten
    )

    # 3. Candidate collection

    all_candidates = []

    # 3A. FAISS semantic search
    

    if vector_db is not None:

        for variant in variants:

            try:
                semantic_results = (
                    vector_db.similarity_search(
                        variant,
                        k=min(
                            24,
                            retrieval_k * 2,
                        ),
                    )
                )

                all_candidates.extend(
                    semantic_results or []
                )

            except Exception:
                # Continue with BM25.
                continue

    # 3B. Per-PDF BM25 search

    for variant in variants:

        for retriever in (
            source_bm25 or {}
        ).values():

            if retriever is None:
                continue

            try:
                lexical_results = (
                    retriever.invoke(
                        variant
                    )
                )

                all_candidates.extend(
                    lexical_results or []
                )

            except Exception:
                continue

    # 4. Deduplicate candidates

    all_candidates = (
        deduplicate_documents(
            all_candidates
        )
    )

    if not all_candidates:
        # Last-resort local fallback: if FAISS/BM25 returned nothing,
        # search the already-indexed chunks directly. This is important for
        # small PDFs and for queries whose wording differs from the PDF.
        # It still remains fully grounded because every returned document
        # comes from the uploaded PDF chunks.
        try:
            fallback_scored = []
            for index, doc in enumerate(chunks):
                try:
                    overlap, coverage, lexical_score = _lexical_score(
                        rewritten,
                        getattr(doc, "page_content", ""),
                    )
                    fallback_scored.append(
                        (lexical_score, overlap, coverage, -index, doc)
                    )
                except Exception:
                    continue

            if fallback_scored:
                fallback_scored.sort(reverse=True)
                fallback = [item[4] for item in fallback_scored[:max(final_k, retrieval_k)]]
                fallback = deduplicate_documents(fallback)
                if fallback:
                    try:
                        fallback = rerank_documents(
                            rewritten, fallback, top_k=final_k
                        ) or fallback
                    except Exception:
                        pass
                    return rewritten, fallback[:final_k]
        except Exception:
            pass

        return rewritten, []

    # 5. Group candidates by PDF

    groups = _source_groups(
        all_candidates
    )

    if not groups:
        return rewritten, []

    # 6. Determine multi-document intent

    multi_document = (
        _is_multi_document_query(
            rewritten
        )
    )

    # 7. Score every PDF

    source_scores = []

    for source, docs in groups.items():

        best_overlap = 0
        best_coverage = 0.0
        best_lexical_score = 0.0

        for doc in docs:

            try:
                overlap, coverage, lexical_score = (
                    _lexical_score(
                        rewritten,
                        getattr(
                            doc,
                            "page_content",
                            "",
                        ),
                    )
                )

                best_overlap = max(
                    best_overlap,
                    overlap,
                )

                best_coverage = max(
                    best_coverage,
                    coverage,
                )

                best_lexical_score = max(
                    best_lexical_score,
                    lexical_score,
                )

            except Exception:
                continue

        # Prefer lexical evidence when it exists, but do not reject a
        # semantically retrieved PDF just because the user used different
        # words or another language (for example, "chutti" vs "leave").
        # FAISS/BM25 already supplied these candidates, so keeping them is
        # still grounded in the uploaded PDFs.
        relevant, _, _ = (
            _document_relevance(
                rewritten,
                docs,
            )
        )

        if not relevant:
            # Keep semantic/lexical candidates as a fallback. They are ranked
            # later by candidate count and original retrieval order.
            source_score = 0.0
        else:
            source_score = (
                best_lexical_score
                + 0.20 * min(
                    best_overlap,
                    10,
                )
            )

        source_scores.append(
            (
                source_score,
                best_overlap,
                best_coverage,
                source,
                docs,
            )
        )

        continue

    # 8. No evidence found

    if not source_scores:
        return rewritten, []

    # 9. Rank PDFs -

    source_scores.sort(
        key=lambda item: (
            item[1],       # lexical overlap
            item[2],       # coverage
            item[0],       # total score
        ),
        reverse=True,
    )

    # 10. Select PDF(s)

    if multi_document:

        selected_sources = (
            source_scores[
                :min(
                    3,
                    len(source_scores),
                )
            ]
        )

    else:

        # Normal question -> strongest PDF only.
        selected_sources = [
            source_scores[0]
        ]

    # 11. Collect selected chunks
    
    selected = []

    allowed_sources = set()

    for (
        _score,
        _overlap,
        _coverage,
        source,
        docs,
    ) in selected_sources:

        allowed_sources.add(
            source
        )

        selected.extend(
            docs
        )

    selected = (
        deduplicate_documents(
            selected
        )
    )

    if not selected:
        return rewritten, []

    # 12. Rerank selected chunks
    
    try:
        ranked = rerank_documents(
            rewritten,
            selected,
            top_k=final_k,
        )

    except Exception:
        ranked = []

    if ranked:

        filtered_ranked = [
            doc
            for doc in ranked
            if _source(doc)
            in allowed_sources
        ]

        ranked = (
            deduplicate_documents(
                filtered_ranked
            )
        )

        if ranked:
            return (
                rewritten,
                ranked[:final_k],
            )

    # 14. Reranker fallback
    
    return (
        rewritten,
        selected[:final_k],
    )