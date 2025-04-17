import os
import json
import logging
from typing import Any, Dict

from langchain.vectorstores.pgvector import PGVector
from langgraph.nodes.retrieve_context import retrieve_context

logger = logging.getLogger(__name__)


def parse_requirements(
    vector_store: PGVector,
    question: str,
    output_path: str = "requirements.json",
    model_name: str = None,
    k: int = 5
) -> Dict[str, Any]:
    """
    Retrieve and parse functional requirements from SRS chunks into a JSON spec file.

    Args:
        vector_store (PGVector): The PGVector store with SRS embeddings.
        question (str): Query to extract requirements.
        output_path (str): File path to write the JSON spec.
        model_name (str, optional): Groq LLM model name; loaded from env if None.
        k (int): Number of top documents to retrieve.

    Returns:
        Dict[str, Any]: Parsed JSON spec dictionary.

    Raises:
        ValueError: If the LLM output does not contain valid JSON.
    """
    logger.info("Calling retrieve_context to extract requirements JSON string...")
    raw_output = retrieve_context(
        vector_store=vector_store,
        question=question,
        model_name=model_name,
        k=k
    )

    # Clean and extract JSON substring
    cleaned = raw_output.strip()
    # Remove markdown code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`\n ")

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        logger.error("No JSON object found in LLM output: %s", raw_output)
        raise ValueError("No JSON object found in LLM output")

    json_text = cleaned[start : end + 1]

    # Parse JSON
    try:
        spec = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON: %s", e)
        logger.debug("Extracted JSON text: %s", json_text)
        raise ValueError(f"Invalid JSON output from retrieve_context: {e}")

    # Write to file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(spec, f, indent=2)
    logger.info("Written requirements spec to %s", output_path)

    return spec


__all__ = ["parse_requirements"]

# import json
# import os
# import re

# def clean_json_from_markdown(text: str) -> str:
#     """
#     Remove markdown code fences and extract clean JSON from LLM output.
#     """
#     # Remove ```json ... ```
#     text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
#     text = text.strip().strip("`")

#     # Extract first valid JSON object in string
#     start = text.find("{")
#     end = text.rfind("}")
#     if start != -1 and end != -1 and end > start:
#         return text[start:end+1]
#     return text

# def parse_requirements(json_str: str, output_path="requirements.json") -> dict:
#     try:
#         cleaned = clean_json_from_markdown(json_str)
#         spec = json.loads(cleaned)
#     except json.JSONDecodeError as e:
#         print("\n--- Raw LLM Output ---\n", json_str)
#         print("\n--- Cleaned Text ---\n", cleaned)
#         raise ValueError(f"Invalid JSON output from retrieve_context: {e}")

#     # Save to file
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(spec, f, indent=4)
#     print(f"✔ Parsed JSON saved to {output_path}")
#     return spec

# if __name__ == "__main__":
#     # Load example LLM output from test file or string
#     with open("sample_response.txt", "r", encoding="utf-8") as f:
#         llm_response = f.read()

#     parsed = parse_requirements(llm_response)
#     print("\n--- Final Parsed Spec ---\n", json.dumps(parsed, indent=2))