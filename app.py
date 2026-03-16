import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb

from config import EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_K, LLM_BASE_URL, LLM_API_KEY, CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION
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

embeddings = OpenAIEmbeddings(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model=EMBEDDING_MODEL
)

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

vectorstore = Chroma(
    client=chroma_client,
    collection_name=CHROMA_COLLECTION,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

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

        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Fallback to Tavily web search if local context is insufficient
        web_context = ""
        if not docs:
            web_context = search_web(query)

        if web_context:
            context = context + "\n\n--- Web Search Results ---\n\n" + web_context if context else web_context

        answer = chain.invoke({"context": context, "question": query})

    st.subheader("Answer")

    st.write(answer)

    st.subheader("Sources")

    for i, doc in enumerate(docs):

        st.markdown(f"**Source {i+1}**")

        st.write(doc.page_content[:400])

    if web_context:
        st.markdown("**+ Additional context from [official documentation website](https://docs.microfocus.com/doc/76/25.2/home)**")

        st.write(doc.page_content[:400])