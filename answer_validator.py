"""Fast deterministic answer validation.

Validates whether the generated answer has usable content.
The main LLM is already instructed to answer only from PDF context,
so this module avoids an additional LLM call.
"""

import re


REFUSAL_MARKERS = {
    "i don't know based on the provided pdf context",
    "i don't know based on the provided pdf",
    "mujhe diye gaye pdf context ki bunyaad par iska jawab maloom nahi",
    "مجھے فراہم کردہ pdf کے سیاق و سباق کی بنیاد پر اس کا جواب معلوم نہیں",
    "لا أعرف الإجابة بناءً على سياق ملف pdf المقدم",
}


ERROR_MARKERS = (
    "local ai model is unavailable",
    "failed to initialize ollama",
    "ollama is unavailable",
    "model is unavailable",
    "model not found",
    "connection refused",
    "connection error",
    "timeout",
    "traceback",
    "exception:",
    "error:",
)


def _normalize(text):
    return re.sub(
        r"\s+",
        " ",
        str(text or "").strip().lower(),
    )


def _is_refusal(answer):
    normalized = _normalize(answer)

    return any(
        marker in normalized
        for marker in REFUSAL_MARKERS
    )


def _is_error(answer):
    normalized = _normalize(answer)

    return any(
        marker in normalized
        for marker in ERROR_MARKERS
    )


def validate_answer(question, context, answer):
    """
    Fast deterministic validation.

    Returns:
        SUPPORTED
        NOT_SUPPORTED
    """

    question = str(question or "").strip()
    context = str(context or "").strip()
    answer = str(answer or "").strip()

    # Missing input.
    if not question or not answer:
        return "NOT_SUPPORTED"

    # Model explicitly refused to answer.
    if _is_refusal(answer):
        return "NOT_SUPPORTED"

    # No retrieved PDF context.
    if not context:
        return "NOT_SUPPORTED"

    # Empty / meaningless answer.
    if len(answer) < 2:
        return "NOT_SUPPORTED"

    # Model/runtime error leaked into the answer.
    if _is_error(answer):
        return "NOT_SUPPORTED"

    # Retrieval + prompt already provide grounding.
    return "SUPPORTED"