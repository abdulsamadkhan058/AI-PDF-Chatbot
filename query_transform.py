"""Deterministic multilingual query transformation."""

import re

# English / Urdu / Arabic characters.
_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+"
    r"|"
    r"[\u0600-\u06FF"
    r"\u0750-\u077F"
    r"\u08A0-\u08FF"
    r"\uFB50-\uFDFF"
    r"\uFE70-\uFEFF]+"
)

def _clean_query(query):
    """Safely normalize the incoming query."""
    try:
        return str(query or "").strip()
    except Exception:
        return ""

def _extract_tokens(query):
    """Extract English, Roman Urdu and Arabic/Urdu tokens."""
    try:
        return _TOKEN_PATTERN.findall(
            query.lower()
        )
    except (
        TypeError,
        AttributeError,
        re.error,
    ):
        return []
    except Exception:
        return []

def _remove_duplicates(tokens):
    """Preserve token order while removing duplicates."""
    try:
        return list(
            dict.fromkeys(
                token
                for token in tokens
                if token
            )
        )
    except Exception:
        return []


def transform_query(query):
    """Create at most two deterministic query variants.

    Variant 1:
        Original user query.

    Variant 2:
        Clean lexical version containing unique
        English/Urdu/Arabic/Roman-Urdu tokens.

    Example:

        "What is my Iqama status?"
        ->
        [
            "What is my Iqama status?",
            "what is my iqama status"
        ]

    If transformation is unnecessary or fails,
    the original query remains available.
    """

    original_query = _clean_query(query)

    if not original_query:
        return []

    try:
        tokens = _extract_tokens(
            original_query
        )

        if not tokens:
            return [original_query]

        unique_tokens = _remove_duplicates(
            tokens
        )

        if not unique_tokens:
            return [original_query]

        compact_query = " ".join(
            unique_tokens
        ).strip()

        variants = [original_query]

        if (
            compact_query
            and compact_query.casefold()
            != original_query.casefold()
        ):
            variants.append(
                compact_query
            )

        # Maximum two variants keeps retrieval
        return variants[:2]

    except (
        TypeError,
        AttributeError,
        ValueError,
        re.error,
    ):
        return [original_query]

    except Exception:
        # Final safety fallback.
        return [original_query]