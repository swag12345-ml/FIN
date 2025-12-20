import streamlit as st  # must be first
import os
import json
import torch
import asyncio
import numpy as np
from dotenv import load_dotenv

# PDF + OCR
import fitz  # PyMuPDF
import easyocr
from pdf2image import convert_from_path

# LangChain (LATEST)
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ------------------------------------------------------------------

st.set_page_config(page_title="Chat with Swag AI", page_icon="📝", layout="centered")
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# ------------------------------------------------------------------
# GROQ KEY
def load_groq_api_key():
    try:
        with open(os.path.join(working_dir, "config.json"), "r") as f:
            return json.load(f).get("GROQ_API_KEY")
    except FileNotFoundError:
        st.error("🚨 config.json not found.")
        st.stop()

groq_api_key = load_groq_api_key()
if not groq_api_key:
    st.error("🚨 GROQ_API_KEY missing.")
    st.stop()

# ------------------------------------------------------------------
# OCR
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        texts = [page.get_text() for page in doc if page.get_text().strip()]
        doc.close()
        return texts if texts else extract_text_from_images(file_path)
    except Exception as e:
        st.error(e)
        return []

def extract_text_from_images(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=5)
        return ["\n".join(reader.readtext(np.array(img), detail=0)) for img in images]
    except Exception as e:
        st.error(e)
        return []

# ------------------------------------------------------------------
# VECTOR STORE
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if DEVICE == "cuda":
        embeddings.client = embeddings.client.to("cuda")

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text("\n".join(documents))

    return FAISS.from_texts(chunks, embeddings)

# ------------------------------------------------------------------
# CHAIN (MEMORY + RETRIEVAL ENABLED)
def create_chain(vectorstore):
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=groq_api_key
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        chain_type="stuff",
        verbose=False
    )

# ------------------------------------------------------------------
# UI
st.title("🦙 Chat with Swag AI (Groq + Memory + RAG)")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    all_text = []

    for file in uploaded_files:
        path = os.path.join(working_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())

        text = extract_text_from_pdf(path)
        all_text.extend(text)
        st.success(f"✅ {file.name} processed")

    if all_text:
        st.session_state.vectorstore = setup_vectorstore(all_text)
        st.session_state.conversation_chain = create_chain(
            st.session_state.vectorstore
        )

# ------------------------------------------------------------------
# CHAT HISTORY DISPLAY
if "memory" in st.session_state:
    for msg in st.session_state.memory.chat_memory.messages:
        with st.chat_message("user" if msg.type == "human" else "assistant"):
            st.markdown(msg.content)

# ------------------------------------------------------------------
# CHAT INPUT
user_input = st.chat_input("Ask your PDFs...")

async def get_response(question):
    result = await asyncio.to_thread(
        st.session_state.conversation_chain.invoke,
        {"question": question}
    )
    return result["answer"]

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        answer = asyncio.run(get_response(user_input))
    except Exception as e:
        answer = str(e)

    with st.chat_message("assistant"):
        st.markdown(answer)
