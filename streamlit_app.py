from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama 
from streamlit_mic_recorder import mic_recorder
import tempfile
import streamlit as st
import whisper
import time
from speak import speak_text

st. set_page_config(page_title="AI PDF Chatbot", page_icon="🤖")
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")
model= load_whisper_model()

@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
@st.cache_resource
def load_llm():
    return ChatOllama(model="mistral", temperature=0)

@st.cache_resource
def create_vector_db(chunks):
    embedding= load_embedding()
    return FAISS.from_documents(documents=chunks,embedding=embedding)

st.title("🤖AI PDF Chatbot")
if "message" not in st.session_state:
    st.session_state.message= []
st.write("Welcome to your AI PDF Chatbot")
st.markdown("---")
with st.sidebar:
    st.header("📂 Upload PDF")
    if st.button("🗑️ Clear Chat"):
        st.session_state.message = []
        st.rerun()
uploaded_files = st.file_uploader("Choose your PDF files",type=["pdf"],accept_multiple_files=True)
if uploaded_files:
    st.success("✅ PDF Uploaded Successfully!")
   
    pdf_paths=[]
    for uploaded_file in uploaded_files:
        st.write("File Name:", uploaded_file.name)
        st.write(" File Size:", uploaded_file.size,"bytes")
        file_path =f"uploads/{uploaded_file.name}"
        with open (file_path, "wb") as f:
             f.write(uploaded_file.getbuffer())
        pdf_paths.append(file_path)
    st.success("PDF Saved Successfully!")
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs=loader.load()
        documents.extend(docs)
    st.success(f"PDF Loaded Successfully! Total Pages:{len(documents)}")
    splitter= RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    st.success(f"Total Chunks:{len(chunks)}")
    
    st.success("✅Embedding Model Loaded Successfully!")
   
    vector_db = create_vector_db(chunks)
    
    st.success("✅ FAISS Database Created Successfully!")
    st.markdown("---")
    for message in st.session_state.message:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query= None

    audio = mic_recorder(start_prompt="🎙️Start Recording",stop_prompt="⏹️ Stop Recording",key="recorder")
    if audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio["bytes"])
                audio_path = f.name
            result=model.transcribe(audio_path)
            query= result["text"]
            st.success(f"You Said:{query}")
    
    text_query = st.chat_input("💬 Ask Question About Your PDF...")
    if text_query:
        query=text_query
    if query:
        with st.chat_message("user"):
            st.markdown(query)

        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        results = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in results])
        source_info = []
        source_pages = []

        for doc in results:
            file_name = doc.metadata.get("source", "").split("/")[-1]
            page_number = doc.metadata.get("page", 0) + 1

            source_info.append(
                {
                    "file": file_name,
                    "page": page_number
                }
            )

            if page_number not in source_pages:
                source_pages.append(page_number)
        history = ""

        for msg in st.session_state.message[-6:]:
            history += f"{msg['role']}: {msg['content']}\n"

        prompt = f"""
                You are an AI Assistant.
                Rules:
                - Answer ONLY from the PDF Context.
                - Use Conversation History when needed.
                - If the user asks "Explain more", "Continue", "Why", "Summarize", etc., use previous conversation.
                - Maximum 3 short sentences.
                - If answer is not found, reply:
                "I don't know based on the provided PDF."
                Conversation History:
                {history}
                PDF Context:
                {context}
                Question:
                {query}
                Answer:
                """
        llm = load_llm()
            
        with st.spinner("🤖 Thinking..."):
            start = time.time()
            full_response = ""
            with st.chat_message("assistant"):
                placeholder = st.empty()
                for chunk in llm.stream(prompt):
                    if chunk.content:
                        full_response += chunk.content
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                st.markdown("### 📄 Sources")
                shown = set()
                for source in source_info:

                    key = (source["file"], source["page"])

                    if key not in shown:
                        st.write(f"📂 {source['file']} | Page {source['page']}")
                        shown.add(key)
            end = time.time()
        speak_text(full_response)
        st.success(f"⏱️ LLM Time: {end - start:.2f} seconds")
        st.session_state.message.append({"role": "user","content": query} )
        st.session_state.message.append( {  "role": "assistant","content": full_response  }  )
        with open("output.mp3", "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")