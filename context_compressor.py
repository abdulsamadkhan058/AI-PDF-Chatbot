"""Deterministic context compression."""

import re


DEFAULT_MAX_CHARS = 7000
MIN_MAX_CHARS = 500


def _safe_page_number(doc):
    try:
        return int(doc.metadata.get("page", 0)) + 1
    except (TypeError, ValueError, AttributeError):
        return 1


def _safe_source_name(doc):
    try:
        source = str(
            doc.metadata.get(
                "source",
                "Unknown",
            )
        )

        return (
            source
            .replace("\\", "/")
            .split("/")[-1]
            or "Unknown"
        )

    except Exception:
        return "Unknown"


def _safe_max_chars(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = DEFAULT_MAX_CHARS

    return max(
        MIN_MAX_CHARS,
        value,
    )


def compress_context(
    query,
    documents,
    max_chars=DEFAULT_MAX_CHARS,
):
    """
    Compress retrieved PDF documents into a compact context.

    Each document keeps:
        SOURCE
        PAGE
        TEXT

    The output never exceeds max_chars by much and safely
    handles malformed documents or metadata.
    """

    if not documents:
        return ""

    max_chars = _safe_max_chars(max_chars)

    blocks = []
    used = 0

    for doc in documents:
        try:
            raw_text = getattr(
                doc,
                "page_content",
                "",
            )

            text = re.sub(
                r"\s+",
                " ",
                str(raw_text or "").strip(),
            )

            if not text:
                continue

            source = _safe_source_name(doc)
            page = _safe_page_number(doc)

            block = (
                f"SOURCE: {source}\n"
                f"PAGE: {page}\n"
                f"{text}"
            )

            separator_size = 2
            required = len(block) + separator_size

            if used + required <= max_chars:
                blocks.append(block)
                used += required
                continue

            remaining = max_chars - used

            # Only add a partial chunk if enough useful text remains.
            if remaining > 300:
                partial = block[:remaining].rstrip()

                if partial:
                    blocks.append(
                        partial + " …"
                    )

            break

        except Exception:
            # One malformed document should not break
            # the complete retrieval/answer pipeline.
            continue

    return "\n\n".join(blocks)