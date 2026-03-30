"""
Pinecone vector store setup and retrieval helpers.

This module is the single place that owns the Pinecone connection so that
every other module can stay decoupled from the SDK details.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import settings


def get_vectorstore() -> PineconeVectorStore:
    """Return a LangChain Pinecone vector store backed by OpenAI embeddings."""
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=settings.pinecone_api_key,
    )


def retrieve(query: str, k: int | None = None) -> list[tuple]:
    """
    Run similarity search against Pinecone.

    Returns a list of (Document, score) tuples, highest score first.
    The score is cosine similarity in [0, 1].
    """
    k = k or settings.top_k
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(query, k=k)
