"""CLI version of AI PDF Bot."""

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from answer_validator import validate_answer
from context_compressor import compress_context
from conversation_memory import ConversationMemory
from core.embeddings import load_embedding
from core.llm import load_llm
from language_utils import detect_language, language_instruction
from retrieval import build_source_bm25, retrieve_documents


REFUSALS = {
    "English": "I don't know based on the provided PDF context.",
    "Urdu": "مجھے فراہم کردہ PDF کے سیاق و سباق کی بنیاد پر اس کا جواب معلوم نہیں۔",
    "Roman Urdu": "Mujhe diye gaye PDF context ki bunyaad par iska jawab maloom nahi.",
    "Arabic": "لا أعرف الإجابة بناءً على سياق ملف PDF المقدم.",
}


def refusal(language):
    return REFUSALS.get(language, REFUSALS["English"])


def load_pdf(path):
    """Load and validate the PDF."""
    try:
        if not path:
            print("Error: PDF path cannot be empty.")
            return None

        if not os.path.isfile(path):
            print(f"Error: PDF not found: {path}")
            return None

        if not path.lower().endswith(".pdf"):
            print("Error: Please provide a PDF file.")
            return None

        print("\nLoading PDF...")

        documents = PyPDFLoader(path).load()

        if not documents:
            print("Error: The PDF contains no readable pages.")
            return None

        print(f"Loaded {len(documents)} page(s).")
        return documents

    except Exception as exc:
        print(f"Error while loading PDF: {exc}")
        return None


def create_chunks(documents):
    """Split PDF documents into searchable chunks."""
    try:
        print("Splitting PDF into chunks...")

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        ).split_documents(documents)

        if not chunks:
            print("Error: No text chunks were created.")
            return None

        print(f"Created {len(chunks)} chunk(s).")
        return chunks

    except Exception as exc:
        print(f"Error while splitting PDF: {exc}")
        return None


def create_vector_database(chunks):
    """Create the FAISS vector database."""
    try:
        print("Creating vector database...")

        embedding = load_embedding()

        if embedding is None:
            print("Error: Embedding model could not be loaded.")
            return None

        database = FAISS.from_documents(
            chunks,
            embedding,
        )

        if database is None:
            print("Error: FAISS database could not be created.")
            return None

        print("FAISS database created successfully.")
        return database

    except Exception as exc:
        print(f"Error while creating FAISS database: {exc}")
        return None


def create_bm25_index(chunks):
    """Create the BM25 keyword search index."""
    try:
        print("Creating BM25 search index...")

        retriever = build_source_bm25(
            chunks,
            8,
        )

        if retriever is None:
            print("Error: BM25 search index could not be created.")
            return None

        print("BM25 index created successfully.")
        return retriever

    except Exception as exc:
        print(f"Error while creating BM25 index: {exc}")
        return None


def create_memory():
    """Initialize conversation memory."""
    try:
        return ConversationMemory()

    except Exception as exc:
        print(f"Error while initializing conversation memory: {exc}")
        return None


def create_llm():
    """Load the configured language model."""
    try:
        print("Loading AI model...")

        llm = load_llm()

        if llm is None:
            print("Error: AI model could not be loaded.")
            return None

        print("AI model loaded successfully.")
        return llm

    except Exception as exc:
        print(f"Error while loading AI model: {exc}")
        print(
            "Make sure Ollama is running and the required model "
            "is installed."
        )
        return None


def build_conversation_history(history):
    """Convert conversation memory into prompt text."""
    try:
        return "\n".join(
            f"User: {item['user']}\n"
            f"Assistant: {item['assistant']}"
            for item in history
        )

    except Exception as exc:
        print(f"Warning: Could not prepare conversation history: {exc}")
        return ""


def build_prompt(query, context, history_text, language):
    """Build a strictly PDF-grounded prompt."""

    try:
        return f"""You are a strictly grounded PDF assistant.

{language_instruction(language)}

RULES:
1. Use ONLY the PDF context.
2. Never use outside knowledge.
3. Never guess or invent information.
4. Every factual claim must be supported by the PDF context.
5. If the answer is not supported by the PDF, respond exactly:
{refusal(language)}
6. Answer in the user's language.
7. Do not switch languages.
8. Do not provide information that is not present in the PDF.

Conversation:
{history_text}

PDF context:
{context}

Question:
{query}

Answer:"""

    except Exception as exc:
        print(f"Error while building prompt: {exc}")
        return None


def retrieve_pdf_context(
    query,
    database,
    bm25,
    chunks,
    history,
):
    """Retrieve relevant PDF documents."""

    try:
        return retrieve_documents(
            query,
            database,
            bm25,
            chunks,
            history,
            retrieval_k=10,
            final_k=6,
        )

    except Exception as exc:
        print(f"Error while searching PDF: {exc}")
        return "", []


def generate_answer(
    llm,
    query,
    context,
    history_text,
    language,
):
    """Generate and validate an LLM answer."""

    prompt = build_prompt(
        query,
        context,
        history_text,
        language,
    )

    if not prompt:
        return refusal(language)

    try:
        response = llm.invoke(prompt)

        if response is None:
            print("Error: AI model returned no response.")
            return refusal(language)

        answer = str(
            getattr(response, "content", response)
        ).strip()

        if not answer:
            print("Error: AI model returned an empty answer.")
            return refusal(language)

    except Exception as exc:
        print(f"AI model error: {exc}")
        print(
            "Check that Ollama is running and the configured "
            "model is available."
        )
        return refusal(language)

    try:
        validation = validate_answer(
            query,
            context,
            answer,
        )

        if validation == "NOT_SUPPORTED":
            return refusal(language)

    except Exception as exc:
        print(f"Answer validation error: {exc}")
        return refusal(language)

    return answer


def display_sources(results):
    """Display unique PDF source pages."""

    print("\nSources:")

    seen = set()

    for document in results:
        try:
            filename = os.path.basename(
                str(
                    document.metadata.get(
                        "source",
                        "Unknown PDF",
                    )
                )
            )

            page = int(
                document.metadata.get(
                    "page",
                    0,
                )
            ) + 1

            key = (
                filename,
                page,
            )

            if key not in seen:
                print(
                    f"- {filename} — Page {page}"
                )
                seen.add(key)

        except Exception as exc:
            print(
                f"- Unable to read source information: {exc}"
            )


def process_query(
    query,
    database,
    bm25,
    chunks,
    memory,
    llm,
):
    """Process one user question."""

    try:
        language = detect_language(query)

    except Exception as exc:
        print(f"Language detection error: {exc}")
        language = "English"

    try:
        history = memory.get_history()

    except Exception as exc:
        print(f"Conversation memory error: {exc}")
        history = []

    rewritten, results = retrieve_pdf_context(
        query,
        database,
        bm25,
        chunks,
        history,
    )

    if not results:
        return refusal(language), []

    try:
        context = compress_context(
            rewritten,
            results,
            7000,
        )

    except Exception as exc:
        print(f"Error while preparing PDF context: {exc}")
        return refusal(language), []

    history_text = build_conversation_history(
        history
    )

    answer = generate_answer(
        llm,
        query,
        context,
        history_text,
        language,
    )

    return answer, results


def main():
    """Main CLI application."""

    print("\n=== AI PDF Bot ===")

    try:
        path = input("PDF path: ").strip()

    except (KeyboardInterrupt, EOFError):
        print("\nApplication closed.")
        return

    documents = load_pdf(path)

    if documents is None:
        return

    chunks = create_chunks(documents)

    if chunks is None:
        return

    database = create_vector_database(chunks)

    if database is None:
        return

    bm25 = create_bm25_index(chunks)

    if bm25 is None:
        return

    memory = create_memory()

    if memory is None:
        return

    llm = create_llm()

    if llm is None:
        return

    print("\n================================")
    print("AI PDF Bot is ready.")
    print("Type your question.")
    print("Type 'exit' to close the application.")
    print("================================")

    while True:

        try:
            query = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n\nAI PDF Bot closed.")
            break

        if not query:
            print("Please enter a question.")
            continue

        if query.lower() == "exit":
            print("\nGoodbye.")
            break

        try:
            answer, results = process_query(
                query,
                database,
                bm25,
                chunks,
                memory,
                llm,
            )

        except Exception as exc:
            print(
                f"Unexpected query processing error: {exc}"
            )
            answer = (
                "An unexpected error occurred "
                "while processing your question."
            )
            results = []

        print("\nBot:")
        print(answer)

        if results:
            display_sources(results)

        try:
            memory.add_message(
                query,
                answer,
            )

        except Exception as exc:
            print(
                f"Warning: Could not save conversation memory: {exc}"
            )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n\nAI PDF Bot stopped by user.")

    except Exception as exc:
        print(
            f"\nUnexpected application error: {exc}"
        )