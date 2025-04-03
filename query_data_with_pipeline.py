# FIXME: This version that using with "pipeline" but not work now.
# Try later if have time.

import argparse
import os
from dotenv import load_dotenv
from transformers import pipeline
from get_embedding import get_embedding
from langchain_chroma import Chroma
import torch
import functools

device = "cuda" if torch.cuda.is_available() else "cpu"

CHROMA_PATH = os.getenv("CHROMA_PATH", "./myDB")
MODEL_DIR = os.getenv("MODEL_DIR", "./model")
MODEL_ID = "google/gemma-3-1b-it"


# Read this page how to use https://huggingface.co/blog/gemma3
@functools.lru_cache(maxsize=1)
def load_gemma_model():
    load_dotenv()
    os.makedirs(MODEL_DIR, exist_ok=True)
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        device=device,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    return pipe


@functools.lru_cache(maxsize=1)
def get_db():
    embedding_model = get_embedding()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)


def query_rag(query_text: str, pipe):
    # Read DB
    db = get_db()

    # search the DB - Retrieval
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    if len(context_text) > 10000:
        context_text = context_text[:10000] + "..."

    # Message format as Gemma 3
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Answer the question based only on the following context:\n{context_text}\n---\n{query_text}",
                }
            ],
        }
    ]

    output = pipe(messages, max_new_tokens=256)
    response_text = output[0]["generated_text"][-1]["content"]

    # PERF: debug
    print(f"\nRESPONE TEXT\n===============================\n{response_text}")

    # Collect sources
    source = [doc.metadata.get("id", None) for doc, _score in results]

    # Summarize the response
    summary_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Summarize the following response in 5 sentences:\n{response_text}",
                }
            ],
        }
    ]
    summary_output = pipe(summary_messages, max_new_tokens=256)
    summary_response = summary_output[0]["generated_text"][-1]["content"]

    formatted_response = (
        f"\n=== Response ===\n{summary_response}\n\n" f"=== Sources ===\n {source}"
    )
    print(formatted_response)
    return response_text


def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    pipe = load_gemma_model()
    query_rag(query_text, pipe)


if __name__ == "__main__":
    main()
