import os
import time
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


PROMPT_TEMPLATE = """Use the following context from the documentation to answer the question.
If the answer is not found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )
    return vectorstore


def main():
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    print("Loading vector database...")
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
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
        print()


if __name__ == "__main__":
    main()
