import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from google import genai


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)

def _get_api_key():
    # Google documents GEMINI_API_KEY and GOOGLE_API_KEY; if both exist,
    # GOOGLE_API_KEY takes precedence. Support both so deployment is flexible.
    key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key or key.lower() in {"your_gemini_api_key_here", "your_api_key_here", "changeme"}:
        return None
    return key


# Public alias so other modules (app.py's chart/vision call) use the exact
# same GOOGLE_API_KEY/GEMINI_API_KEY resolution as normal text answers,
# instead of duplicating (and risking drifting from) this lookup.
get_api_key = _get_api_key


class GeminiChatModel(BaseChatModel):

    @property
    def _llm_type(self) -> str:
        return "gemini-interactions"

    def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ):
        api_key = _get_api_key()

        if not api_key:
            raise RuntimeError(
                "Gemini API key is not configured. Add GEMINI_API_KEY (or GOOGLE_API_KEY) to the project .env file, then restart Chainlit."
            )

        client = genai.Client(api_key=api_key)

        prompt = self._messages_to_text(messages)

        model = (
            os.getenv("AI_PDF_GEMINI_MODEL", "gemini-3.6-flash").strip()
            or "gemini-3.6-flash"
        )
        interaction = client.interactions.create(
            model=model,
            input=prompt,
        )

        text = interaction.output_text or ""

        message = AIMessage(content=text)

        generation = ChatGeneration(message=message)

        return ChatResult(
            generations=[generation]
        )

    def _messages_to_text(self, messages):
        parts = []

        for message in messages:

            if isinstance(message, HumanMessage):
                role = "User"

            else:
                role = "Assistant"

            parts.append(
                f"{role}: {message.content}"
            )

        return "\n\n".join(parts)

    def stream(
        self,
        input,
        config=None,
        *,
        stop=None,
        **kwargs,
    ):
        """
        Simple streaming-compatible interface.

        Gemini Interactions API response is collected first,
        then emitted as one AIMessage chunk.
        """

        yield self.invoke(
            input,
            config=config,
            stop=stop,
            **kwargs,
        )


@lru_cache(maxsize=1)
def load_llm():
    api_key = _get_api_key()

    if not api_key:
        raise RuntimeError(
            "Gemini API key is not configured. Add GEMINI_API_KEY (or GOOGLE_API_KEY) to the project .env file, then restart Chainlit."
        )

    return GeminiChatModel()