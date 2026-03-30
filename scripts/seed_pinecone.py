"""
One-time script: embed and upsert parking documents into Pinecone.

Run once before starting the app:
    python scripts/seed_pinecone.py

The script is idempotent: re-running it overwrites existing vectors
with the same IDs rather than duplicating them.
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env before importing config so env vars are available to langchain_pinecone
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from config import settings
from data.parking_documents import PARKING_DOCUMENTS


def create_index_if_needed(pc: Pinecone) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        print(f"Creating index '{settings.pinecone_index_name}'...")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=1536,          # text-embedding-3-small dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Index '{settings.pinecone_index_name}' already exists.")


def seed() -> None:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    create_index_if_needed(pc)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )

    # langchain_pinecone reads PINECONE_API_KEY from env (set via load_dotenv above)
    print(f"Upserting {len(PARKING_DOCUMENTS)} documents...")
    pinecone_index = pc.Index(settings.pinecone_index_name)
    PineconeVectorStore.from_documents(
        documents=PARKING_DOCUMENTS,
        embedding=embeddings,
        index_name=settings.pinecone_index_name,
    )
    print("Seeding complete.")


if __name__ == "__main__":
    seed()
