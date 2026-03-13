import os
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma


def load_documents():
    """
    Load documentation PDFs
    """
    print("Loading documentation...")

    loader = PyPDFLoader("docs/Product_overview.pdf")
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    return documents


def split_documents(documents):
    """
    Split documents into chunks
    """

    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks


def create_vectorstore(chunks):
    """
    Create embeddings and store in Chroma DB
    """

    print("Creating embeddings...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    print("Building vector database...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )

    print("Vector database saved successfully")

    return vectorstore


def test_search(vectorstore):
    """
    Test similarity search
    """

    print("\nTesting retrieval...")

    query = "How do I configure OAuth authentication?"

    results = vectorstore.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        print(f"\nResult {i+1}")
        print("-" * 50)
        print(doc.page_content[:500])


def main():

    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    documents = load_documents()

    chunks = split_documents(documents)

    vectorstore = create_vectorstore(chunks)

    test_search(vectorstore)


if __name__ == "__main__":
    main()