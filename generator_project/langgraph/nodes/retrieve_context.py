import os
import logging
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.vectorstores.pgvector import PGVector

logger = logging.getLogger(__name__)


def retrieve_context(
    vector_store: PGVector,
    question: str,
    model_name: str = "llama-3.1-8b-instant",
    k: int = 5
) -> str:
    """
    Use a Groq-powered retrieval QA chain to extract functional requirements from SRS chunks.

    Args:
        vector_store (PGVector): Initialized PGVector store with SRS embeddings.
        question (str): The query to extract requirements, e.g., "Extract the functional requirements...".
        model_name (str): Name of the Groq LLM model to use.
        k (int): Number of top documents to retrieve.

    Returns:
        str: JSON-formatted requirements as returned by the LLM.
    """
    # Load environment (for GROQ_API_KEY)
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in environment")

    # Wrap vector store as retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # Prompt for extracting requirements
    template = """
    You are an assistant that extracts the functional requirements from an SRS document of a project.
    Your task is to extract the required points from the given context.
    Given the context (pulled from the SRS):

    {context}

    Please extract and return **only** the following requirements in JSON format with these keys:
    - "endpoints": list of objects {{ "path": "", "method": "", "params":[...], "description": "" }}
    - "logic": description of the system's business rules and computations
    - "schema": description of tables, relationships, and constraints
    - "auth": description of authentication and authorization mechanisms
    """
    prompt = PromptTemplate(input_variables=["context"], template=template)

    # Initialize Groq LLM
    groq_llm = ChatGroq(
        model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key,
    )

    # Build the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=groq_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )

    # Run the chain
    logger.info("Running retrieval QA chain for question: %s", question)
    response = qa_chain.run(question)
    logger.info("Retrieval QA chain completed")
    return response


__all__ = ["retrieve_context"]


# import os
# from dotenv import load_dotenv
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq

# # Load environment variables
# load_dotenv()

# def get_retriever(vector_store, k=5):
#     return vector_store.as_retriever(search_kwargs={"k": k})

# def get_prompt():
#     template = """
# You are an assistant that extracts the functional requirements from an SRS document of a project.
# Your task is to extract the required points from the given context.
# Given the context (pulled from the SRS):

# {context}

# Please extract and return **only** the following requirements in JSON format with these keys:
# - "endpoints": list of objects {{ "path": "", "method": "", "params":[...], "description": "" }}
# - "logic": description of the system's business rules and computations
# - "schema": description of tables, relationships, and constraints
# - "auth": description of authentication and authorization mechanisms
# """
#     return PromptTemplate(input_variables=["context"], template=template)

# def get_llm():
#     return ChatGroq(
#         model=os.getenv("GROQ_MODEL_NAME", "mixtral-8x7b-32768"),  # Use Mixtral as a fallback
#         temperature=0,
#         max_tokens=2048,
#         timeout=60,
#         api_key=os.getenv("GROQ_API_KEY"),
#     )

# def run_retrieval(vector_store, query):
#     retriever = get_retriever(vector_store)
#     prompt = get_prompt()
#     llm = get_llm()

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=False,
#         chain_type_kwargs={"prompt": prompt},
#     )

#     result = qa_chain.run(query)
#     if not result or not result.strip():
#         raise ValueError("LLM output is empty. Check GROQ_API_KEY and model name.")
#     return result

# if __name__ == "__main__":
#     from ingest_srs import create_pgvector_store, load_document, split_document

#     doc_path = "./data/Python SRS.docx"
#     conn_str = os.getenv("PGVECTOR_CONN_STRING")
#     document = load_document(doc_path)
#     chunks = split_document(document)
#     vector_store = create_pgvector_store(chunks, conn_str)

#     output = run_retrieval(vector_store, "Extract the functional requirements from the document.")
#     print("\n--- LLM Response ---\n", output)