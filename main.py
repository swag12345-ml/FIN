import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
import re
import json

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="AI Financial Analyzer", layout="wide")

llm = ChatGroq(
    model="llama3-70b-8192",
    groq_api_key=st.secrets["GROQ_API_KEY"]
)

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

# ---------------- UI ---------------- #
st.title("💰 AI Financial Statement Analyzer")

uploaded_file = st.file_uploader("Upload Bank Statement PDF", type=["pdf"])

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)

    df = extract_transactions(raw_text)
    df["Category"] = df["Description"].apply(categorize)

    st.subheader("📄 Extracted Transactions")
    st.dataframe(df)

    # ---------------- VISUAL INSIGHTS ---------------- #
    st.subheader("📊 Spending Insights")

    expense_df = df[df["Amount"] < 0]
    category_summary = expense_df.groupby("Category")["Amount"].sum().abs()

    fig, ax = plt.subplots()
    category_summary.plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # ---------------- AI INSIGHT PREVIEW ---------------- #
    st.subheader("🤖 AI Spending Analysis")

    summary_text = f"""
    Here is the spending data by category:
    {category_summary.to_dict()}
    """

    ai_prompt = f"""
    Analyze the user's spending habits.
    Tell where they spend the most and least.
    Give 3 improvement tips.
    """

    ai_response = llm.invoke(ai_prompt + summary_text)
    st.write(ai_response.content)

    # ---------------- RAG CHATBOT ---------------- #
    st.subheader("💬 Ask Questions About Your Statement")

    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question")

    if query:
        result = qa_chain({
            "question": query,
            "chat_history": st.session_state.chat_history
        })

        st.session_state.chat_history.append((query, result["answer"]))
        st.write("🤖", result["answer"])
