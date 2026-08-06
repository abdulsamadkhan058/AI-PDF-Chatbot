import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
# from langchain.memory import ConversationBufferMemory
pdf_path = input("Enter PDF Path:")
loader = PyPDFLoader(pdf_path)
documents= loader.load()
print("total pages:", len(documents))
# print("="*40)
# print(documents[0].page_content)
# print("=" *40)
# print(documents[0].metadata)
splitter = RecursiveCharacterTextSplitter( chunk_size=200, chunk_overlap=50)
chunks=splitter.split_documents(documents)
print("Total Chunks:", len(chunks))
from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedded Model Load Successfully")
from langchain_community.vectorstores import FAISS
if os.path.exists("faiss_db"):
    print("Loading Existing FAISS Database...")
    vector_db = FAISS.load_local("faiss_db",embedding, allow_dangerous_deserialization=True)
else:
    print("Creating New FAISS Database...")
    vector_db=FAISS.from_documents(documents=chunks, embedding=embedding)
    vector_db.save_local("faiss_db")
print("FAISS Ready!")

retriever=vector_db.as_retriever(search_kwargs={"k":3})

llm= ChatOllama(model="mistral",temperature=0)
print("LLM loaded Succesfully!")
# memory= ConversationBufferMemory(return_message=True)

print("\n======PDF CHATBOT======")

while True:

    query = input("\nYou: ")
    if query.lower() == "exit":
        print("Bot: Good Bye")
        break
    results=retriever.invoke(query)

    if len(results) == 0:
        print("Bot: I couldn't find anything in the PDF. \n")
        continue

    context = "\n\n".join([doc.page_content for doc in results])
    prompt= f"""
    You are an AI assistant.
    Answer ONLY from the context below.
    If the answer is not present in the content, reply:
    "I don't know based on the context provided PDF."
    Context:
    {context}
    Question:
    {query}
    Answer:
    """
    response = llm.invoke(prompt)
    print("\n Bot: \n")
    print(response.content)

    print("\n" + "_" * 50 + "\n")
