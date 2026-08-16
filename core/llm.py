import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from google import genai

try:
    import streamlit as st
except ImportError:
    st = None


load_dotenv()


def _cache(func):
    if st is not None:
        return st.cache_resource(show_spinner=False)(func)
    return lru_cache(maxsize=1)(func)


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
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found in .env"
            )

        client = genai.Client(api_key=api_key)

        prompt = self._messages_to_text(messages)

        interaction = client.interactions.create(
            model="gemini-3.5-flash",
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

        result = self.invoke(
            input,
            config=config,
            stop=stop,
            **kwargs,
        )

        yield result


@_cache
def load_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found in .env"
        )

    return GeminiChatModel()