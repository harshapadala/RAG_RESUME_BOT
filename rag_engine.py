import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

# 🔥 Dynamic model (change from .env if needed)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Cache embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def build_vectorstore(pdf_path: str) -> FAISS:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    if not docs:
        raise ValueError("No content found in PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("Could not split PDF.")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY missing in .env")

    print("USING MODEL:", GROQ_MODEL)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=GROQ_MODEL,
        temperature=0.2
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return chain


def ask(chain: RetrievalQA, question: str) -> dict:
    if not question.strip():
        return {"answer": "Please ask a valid question.", "sources": []}

    try:
        result = chain.invoke({"query": question})

        return {
            "answer": result.get("result", "No answer found."),
            "sources": result.get("source_documents", [])
        }

    except Exception as e:
        return {
            "answer": f"❌ Error: {str(e)}",
            "sources": []
        }