import os
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb

from config import EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_K, LLM_BASE_URL, LLM_API_KEY, CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION
from web_search import search_web


PROMPT_TEMPLATE = """Use the following context from the documentation to answer the question.
If the answer is not found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


def load_vectorstore():
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
    return vectorstore


def main():
    load_dotenv()

    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY not found in environment variables")

    print("Loading vector database...")
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    print("\nReady! Ask questions about the documentation. Type 'quit' to exit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        # Retrieve documents once and reuse for both answer and sources
        for attempt in range(3):
            try:
                docs = retriever.invoke(question)
                context = "\n\n".join(doc.page_content for doc in docs)

                # Fallback to Tavily web search if local context is insufficient
                web_context = ""
                if not docs:
                    web_context = search_web(question)

                if web_context:
                    context = context + "\n\n--- Web Search Results ---\n\n" + web_context if context else web_context

                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({"context": context, "question": question})
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"Rate limited. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        print(f"\nAnswer: {answer}\n")

        print("Sources:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            print(f"  {i}. {source} (page {page})")
        if web_context:
            print("  + Additional context from official documentation website")
        print()


if __name__ == "__main__":
    main()
