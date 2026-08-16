"""Deterministic multilingual follow-up query rewriting.

Supports:
- English
- Urdu
- Roman Urdu
- Arabic
- Mixed-language queries

No LLM call is made.
The function never raises an exception to the retrieval pipeline.
"""

import re


# Words that commonly indicate a follow-up/reference
_FOLLOW_UP_TERMS = {
    # English
    "this",
    "that",
    "these",
    "those",
    "it",
    "they",
    "them",
    "he",
    "she",

    # Roman Urdu
    "ye",
    "yeh",
    "yah",
    "woh",
    "wo",
    "isko",
    "isey",
    "ise",
    "usko",
    "usey",
    "use",
    "iska",
    "iski",
    "iske",
    "uska",
    "uski",
    "uske",
    "in",
    "un",
    "inhain",
    "unhain",
    "yehi",
    "wohi",

    # Urdu
    "یہ",
    "یہی",
    "وہ",
    "وہی",
    "اس",
    "اسے",
    "اسکا",
    "اسکی",
    "اسکے",
    "ان",
    "انہیں",
    "انکا",
    "انکی",
    "انکے",

    # Arabic
    "هذا",
    "هذه",
    "ذلك",
    "تلك",
    "هؤلاء",
    "هو",
    "هي",
    "هم",
    "ها",
    "هذا",
}


_FOLLOW_UP_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(term)
        for term in sorted(
            _FOLLOW_UP_TERMS,
            key=len,
            reverse=True,
        )
        if re.match(r"^[A-Za-z0-9]+$", term)
    )
    + r")\b"
    r"|"
    + "|".join(
        re.escape(term)
        for term in sorted(
            _FOLLOW_UP_TERMS,
            key=len,
            reverse=True,
        )
        if not re.match(r"^[A-Za-z0-9]+$", term)
    ),
    re.IGNORECASE,
)


def _clean_text(value):
    """Safely convert a value to clean text."""
    try:
        return str(value or "").strip()
    except Exception:
        return ""


def _valid_history(history):
    """Return True only for supported history structures."""
    return isinstance(history, (list, tuple)) and bool(history)


def _has_follow_up_reference(query):
    """Detect whether the query likely refers to previous context."""
    try:
        return bool(
            _FOLLOW_UP_PATTERN.search(
                query
            )
        )
    except (TypeError, AttributeError, re.error):
        return False
    except Exception:
        return False


def rewrite_query(query, history):
    """Rewrite a multilingual follow-up query.

    The previous user question and answer are included only when:
    1. The current query is non-empty.
    2. Valid conversation history exists.
    3. The query contains a likely follow-up/reference term.
    4. A previous user question is available.

    Otherwise the original query is returned unchanged.

    This function is deterministic and never raises an exception.
    """

    original_query = _clean_text(query)

    if not original_query:
        return ""

    try:
        if not _valid_history(history):
            return original_query

        if not _has_follow_up_reference(
            original_query
        ):
            return original_query

        last = history[-1]

        if not isinstance(last, dict):
            return original_query

        previous_user = _clean_text(
            last.get("user")
        )

        previous_answer = _clean_text(
            last.get("assistant")
        )

        if not previous_user:
            return original_query

        # Keep the previous context bounded.
        previous_user = previous_user[:1500]
        previous_answer = previous_answer[:2500]

        if previous_answer:
            return (
                f"Previous question: {previous_user}\n"
                f"Previous answer: {previous_answer}\n"
                f"Current follow-up: {original_query}"
            )

        return (
            f"Previous question: {previous_user}\n"
            f"Current follow-up: {original_query}"
        )

    except (
        TypeError,
        AttributeError,
        IndexError,
        KeyError,
        ValueError,
    ):
        return original_query

    except Exception:
        # Final safety boundary.
        return original_query