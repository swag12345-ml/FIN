# ---------------- IMPORTS ---------------- #
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
import re

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="AI Financial Analyzer", layout="wide")
st.title("💰 AI Financial Statement Analyzer")

# ---------------- PDF TEXT EXTRACTION ---------------- #
def extract_text_from_pdf(pdf):
    doc = fitz.open(stream=pdf.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ---------------- TRANSACTION PARSER ---------------- #
def extract_transactions(text):
    lines = text.split("\n")
    data = []
    for line in lines:
        match = re.search(r"(\d{2}/\d{2}/\d{4}).*?(-?\d+\.\d{2})", line)
        if match:
            date = match.group(1)
            amount = float(match.group(2))
            description = line
            data.append([date, description, amount])
    df = pd.DataFrame(data, columns=["Date", "Description", "Amount"])
    return df

# ---------------- CATEGORY CLASSIFIER ---------------- #
def categorize(desc):
    desc = desc.lower()
    if "amazon" in desc or "flipkart" in desc:
        return "Shopping"
    if "uber" in desc or "ola" in desc:
        return "Travel"
    if "zomato" in desc or "swiggy" in desc:
        return "Food"
    if "rent" in desc or "electricity" in desc:
        return "Bills"
    return "Others"

# ---------------- FILE UPLOAD ---------------- #
uploaded_file = st.file_uploader("Upload Bank Statement PDF", type=["pdf"])

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)

    # ---------------- TRANSACTION DATAFRAME ---------------- #
    df = extract_transactions(raw_text)
    df["Category"] = df["Description"].apply(categorize)
    st.subheader("📄 Extracted Transactions")
    st.dataframe(df)

    # ---------------- VISUAL INSIGHTS ---------------- #
    st.subheader("📊 Spending Insights")
    expense_df = df[df["Amount"] < 0]
    category_summary = expense_df.groupby("Category")["Amount"].sum().abs()
    category_summary = pd.to_numeric(category_summary, errors='coerce').dropna()

    if not category_summary.empty:
        fig, ax = plt.subplots()
        category_summary.plot(kind="bar", ax=ax, color="skyblue")
        ax.set_ylabel("Amount Spent")
        ax.set_title("Spending by Category")
        st.pyplot(fig)
    else:
        st.info("No numeric expense data available for plotting.")

    # ---------------- SIMPLE RAG QA (WITHOUT LLM) ---------------- #
    st.subheader("💬 Ask Questions About Your Statement")

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    # Create vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question")
    if query:
        # Simple similarity search
        results = vectorstore.similarity_search(query, k=3)
        answer = "\n---\n".join(results)
        st.session_state.chat_history.append((query, answer))
        st.write("🤖", answer)
