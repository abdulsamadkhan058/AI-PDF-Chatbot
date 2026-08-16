from collections import deque


DEFAULT_MAX_TURNS = 6


def _safe_max_turns(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = DEFAULT_MAX_TURNS

    return max(1, value)


class ConversationMemory:
    def __init__(self, max_turns=DEFAULT_MAX_TURNS):
        self.history = deque(
            maxlen=_safe_max_turns(max_turns)
        )

    def add_message(self, user_query, assistant_answer):
        self.history.append({
            "user": str(user_query or ""),
            "assistant": str(assistant_answer or ""),
        })

    def get_history(self):
        return list(self.history)

    def clear(self):
        self.history.clear()