# NOTE: This version that using with LLM-only so will use Gemma3ForCausalLM class.
# Gemma3ForCausalLM should be paried with AutoTokenizer
# Read more @ https://huggingface.co/blog/gemma3

import argparse
import os
from tokenize import tokenize
from dotenv import load_dotenv
from langchain_core import messages
from transformers import AutoTokenizer, Gemma3ForCausalLM
from get_embedding import get_embedding
from langchain_chroma import Chroma
import torch
import functools


CHROMA_PATH = os.getenv("CHROMA_PATH", "./myDB")
MODEL_DIR = os.getenv("MODEL_DIR", "./model")
MODEL_ID = "google/gemma-3-4b-it"


# Read this page how to use https://huggingface.co/blog/gemma3
@functools.lru_cache(maxsize=1)
def load_gemma_model():
    load_dotenv()
    os.makedirs(MODEL_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = Gemma3ForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).cuda()  # pyright: ignore

    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    return model, tokenizer


@functools.lru_cache(maxsize=1)
def get_db():
    """
    Get the vector database with caching
    """
    embedding_model = get_embedding()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)


def generate_text(model, tokenizer, prompt, max_new_tokens=256):
    """Generate text using Gemma-3 model and tokenizer"""
    try:
        # Format for Gemma-3 chat format
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            # Generate output
            input_len = inputs["input_ids"].shape[-1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )

            # Decode and return only the new tokens
            response = tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )
        else:
            # Fallback to standard generation if chat template is not available
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                model.device
            )
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
            response = tokenizer.decode(
                outputs[0][input_ids.shape[1] :], skip_special_tokens=True
            )

        return response
    except Exception as e:
        print(f"Error generating text: {e}")
        return f"Error generating response: {str(e)}"


def query_rag(query_text: str, model_tokenizer):
    """
    Perform RAG query and summarization
    """
    model, tokenizer = model_tokenizer
    # Read DB
    db = get_db()

    # search the DB - Retrieval
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    if len(context_text) > 10000:
        context_text = context_text[:10000] + "..."

    # Create prompt
    prompt = f"Answer the question based only on the following context:\n{context_text}\n---\n{query_text}"

    # Generate response
    response_text = generate_text(model, tokenizer, prompt)

    # PERF: debug
    print(f"\nRESPONE TEXT\n===============================\n{response_text}")

    # Collect sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    # Generate summary

    summary_prompt = (
        f"Summarize the following response clearly and concisely. "
        f"Use bullet points for long sentences and important ideas:\n\n{response_text}"
    )

    summary_response = generate_text(model, tokenizer, summary_prompt)

    # Format and print the final response
    formatted_response = (
        f"\n=== Response ===\n{summary_response.strip()}\n\n=== Sources ===\n{sources}"
    )

    print(formatted_response)

    return response_text


def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    model_tokenizer = load_gemma_model()
    query_rag(query_text, model_tokenizer)


if __name__ == "__main__":
    main()
