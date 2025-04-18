
                                             # LOADS AND CHUNKS THE SRS

# -- Takes the uploaded SRS document
# -- Reads it content
# -- Splits into smaller text chunks
# -- Embeds each chunk using an embedding model
# -- Stores the embeddings in a vector database

import os
import logging

from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector

logger = logging.getLogger(__name__)


def ingest_srs(
    document_path: str,
    connection_string: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    collection_name: str = "srs_documents"
) -> PGVector:
    """
    Load an SRS document from disk, split it into text chunks, and store embeddings in a PGVector collection.

    Args:
        document_path (str): Path to the .txt, .pdf, or .docx SRS file.
        connection_string (str): SQLAlchemy connection string for the Postgres + pgvector database.
        chunk_size (int): Maximum number of characters per chunk (default 500).
        chunk_overlap (int): Number of overlapping characters between chunks (default 100).
        collection_name (str): Name of the PGVector collection (default "srs_documents").

    Returns:
        PGVector: The vector store instance containing the uploaded embeddings.
    """

    # Helps loading the document

    ext = os.path.splitext(document_path)[1].lower()
    if ext == ".txt":
        loader = TextLoader(document_path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(document_path)
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(document_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    logger.info(f"Loading document from %s", document_path)
    documents = loader.load()
    logger.info("Loaded %d pages/sections", len(documents))

    # Split into chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    logger.info(
        "Splitting document into chunks (size=%d, overlap=%d)",
        chunk_size, chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    logger.info("Generated %d chunks", len(chunks))

    # Create embeddings and vector store
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    logger.info("Creating PGVector store '%s' at %s", collection_name, connection_string)
    vector_store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=connection_string
    )
    logger.info("PGVector store created successfully")

    return vector_store
