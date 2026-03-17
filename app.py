import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_BASE_URL, LLM_API_KEY
from retriever import create_retriever
from web_search import search_web

load_dotenv()

if not LLM_API_KEY:
    raise ValueError("LLM_API_KEY not found in environment variables")

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

    st.write(f"Model: {LLM_MODEL}")

st.title("🤖 Enterprise Documentation AI Assistant")

st.markdown(
"""
Ask questions about the product documentation.
The assistant retrieves relevant sections and generates answers using RAG.
"""
)

retriever = create_retriever()

llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE
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

        print(f"[QUERY] '{query}'")

        docs = retriever.invoke(query)
        print(f"[RETRIEVER] {len(docs)} chunks retrieved from ChromaDB")
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', '?')
            print(f"  chunk {i+1}: {source} (page {page}) [{len(doc.page_content)} chars]")

        context = "\n\n".join(doc.page_content for doc in docs)

        # Fallback to Tavily web search if local context is insufficient
        web_context = ""
        if not docs:
            print("[TAVILY] No local results — triggering web search fallback")
            web_context = search_web(query)
        else:
            print("[TAVILY] Skipped — local results sufficient")

        if web_context:
            context = context + "\n\n--- Web Search Results ---\n\n" + web_context if context else web_context

        print(f"[LLM] Sending to {LLM_MODEL} ({len(context)} chars context)")
        answer = chain.invoke({"context": context, "question": query})
        print(f"[LLM] Response received ({len(answer)} chars)")

    st.subheader("Answer")

    st.write(answer)

    st.subheader("Sources")

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section", "")
        header = f"**Source {i+1}** \u2014 {source} (page {page})"
        if section:
            header += f" \u2014 {section}"
        st.markdown(header)
        st.write(doc.page_content[:400])

    if web_context:
        st.markdown("**+ Additional context from [official documentation website](https://docs.microfocus.com/doc/76/25.2/home)**")