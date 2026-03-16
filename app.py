import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found")

st.set_page_config(
    page_title="Enterprise Documentation AI Assistant",
    page_icon="🤖",
    layout="wide"
)

with st.sidebar:

    st.header("About")

    st.write(
        """
        This assistant answers questions from enterprise product documentation
        using Retrieval-Augmented Generation (RAG).
        """
    )

    st.write("Model: Gemini 2.5 Flash")

st.title("🤖 Enterprise Documentation AI Assistant")

st.markdown(
"""
Ask questions about the product documentation.
The assistant retrieves relevant sections and answers using Gemini.
"""
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_template(
    """Use the following context from the documentation to answer the question.
If the answer is not found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
)

chain = prompt | llm | StrOutputParser()

query = st.text_input(
    "Ask a question about the documentation:",
    placeholder="Example: How do I configure OAuth authentication?"
)

if query:

    with st.spinner("Searching documentation..."):

        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        answer = chain.invoke({"context": context, "question": query})

    st.subheader("Answer")

    st.write(answer)

    st.subheader("Sources")

    for i, doc in enumerate(docs):

        st.markdown(f"**Source {i+1}**")

        st.write(doc.page_content[:400])