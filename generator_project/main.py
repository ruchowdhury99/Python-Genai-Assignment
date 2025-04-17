# generator_project/main.py

import os
from dotenv import load_dotenv

from langgraph.nodes.ingest_srs import ingest_srs
from langgraph.nodes.retrieve_context import retrieve_context

if __name__ == "__main__":
    # 1) Load env
    load_dotenv()

    # 2) Ingest the SRS document
    document_path = "./data/Python SRS.docx"
    db_uri = os.getenv("PGVECTOR_DB_URI")
    if not db_uri:
        raise RuntimeError("PGVECTOR_DB_URI not set in .env")

    print("▶️  Ingesting SRS into PGVector…")
    vector_store = ingest_srs(
        document_path=document_path,
        connection_string=db_uri,
        collection_name="srs_documents"
    )
    print("✅ Ingestion complete.\n")

    # 3) Retrieve & extract requirements
    question = (
        "Extract the functional requirements from the document "
        "for upcoming project."
    )
    print("▶️  Running retrieval+Groq QA chain…\n")
    result_json = retrieve_context(
        vector_store=vector_store,
        question=question,
        model_name="llama-3.1-8b-instant",  # or your actual model
        k=5
    )

    print("✅ Retrieval complete. Output JSON:\n")
    print(result_json)

# import os
# from dotenv import load_dotenv

# from langgraph.nodes.ingest_srs import ingest_srs
# from langgraph.nodes.retrieve_context import retrieve_context
# from langgraph.nodes.parse_requirements import parse_requirements

# if __name__ == "__main__":
#     # Load environment variables from .env
#     load_dotenv()

#     # 1️⃣ Ingest SRS into PGVector
#     document_path = "./data/Python SRS.docx"
#     db_uri = os.getenv("PGVECTOR_DB_URI")
#     if not db_uri:
#         raise RuntimeError("PGVECTOR_DB_URI not set in .env")

#     print("▶️  Ingesting SRS into PGVector…")
#     vector_store = ingest_srs(
#         document_path=document_path,
#         connection_string=db_uri,
#         collection_name="srs_documents"
#     )
#     print("✅ Ingestion complete.\n")

#     # 2️⃣ (Optional) Quick sanity check via retrieval — you can skip if you only want the JSON spec
#     question = "Extract the functional requirements from the document for upcoming project."
#     print("▶️  Running retrieval+Groq QA chain…")
#     interim_json = retrieve_context(
#         vector_store=vector_store,
#         question=question,
#         model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant"),
#         k=5
#     )
#     print("✅ Retrieval QA output (raw JSON):\n", interim_json, "\n")

#     # 3️⃣ Parse and write out requirements.json
#     output_path = "./requirements.json"
#     print(f"▶️  Parsing requirements and writing to {output_path}…")
#     spec = parse_requirements(
#         vector_store=vector_store,
#         question=question,
#         output_path=output_path,
#         model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant"),
#         k=5
#     )
#     print("✅ requirements.json generated with content:\n", spec)
