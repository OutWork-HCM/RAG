import random
import os
import pandas as pd
import torch
import numpy as np
from time import perf_counter as timer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import textwrap

# Start timing the entire process
start_time = timer()

# Set device to CPU if GPU memory is low, otherwise use GPU if available
gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
gpu_memory_gb = round(gpu_memory_bytes / (2**30))
device = "cpu" if gpu_memory_gb <= 5 else "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA if running on CPU

# -------------------
# Data Loading
# -------------------
# Load precomputed embeddings and text chunks from CSV
text_chunks_and_embedding_df = pd.read_csv("./text_chunks_and_embeddings_df.csv")

# Convert embedding column from string to numpy array
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df[
    "embedding"
].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

# Convert DataFrame to list of dictionaries for easier access
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

# Convert embeddings to PyTorch tensor and move to selected device
embeddings = torch.tensor(
    np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32
).to(device)

# Load sentence transformer model for embedding queries
embedding_model = SentenceTransformer(
    model_name_or_path="all-mpnet-base-v2", device=device
)


# -------------------
# Utility Functions
# -------------------
def print_wrapped(text: str, wrap_length: int = 80) -> None:
    """
    Prints text wrapped at a specified length for better readability.

    Args:
        text: The text to print
        wrap_length: Maximum length of each line (default: 80 characters)
    """
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


# -------------------
# Retrieval Functions
# -------------------
def retrieve_relevant_resources(
    query: str,
    embeddings: torch.Tensor,
    model: SentenceTransformer = embedding_model,
    n_resources_to_return: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieves the top-k most relevant resources for a given query using cosine similarity.

    Args:
        query: The user's query string
        embeddings: Precomputed embeddings of text chunks
        model: Sentence transformer model for embedding the query
        n_resources_to_return: Number of top resources to return

    Returns:
        Tuple of (scores, indices) where scores are similarity scores and indices are positions in pages_and_chunks
    """
    # Embed the query into a vector
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute dot product (cosine similarity) between query embedding and chunk embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]  # pyright: ignore
    end_time = timer()

    # Print retrieval time
    print(f"[INFO] Retrieval time: {end_time - start_time:.5f} seconds")

    # Get top-k scores and indices
    scores, indices = torch.topk(dot_scores, k=n_resources_to_return)
    return scores, indices


# -------------------
# Prompt Formatting
# -------------------
def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """
    Formats a prompt by combining the query with relevant context items for the LLM.

    Args:
        query: The user's query string
        context_items: List of dictionaries containing relevant text chunks

    Returns:
        Formatted prompt string ready for LLM input
    """
    # Combine context items into a bulleted list
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Base prompt with instructions for the LLM
    base_prompt = """Based on the following context items, please answer the query.
Make sure your answers are explanatory and based on the provided context.
\nNow use the following context items to answer the user query:
{context}
\nUser query: {query}
Answer:"""

    # Fill in context and query into the base prompt
    formatted_prompt = base_prompt.format(context=context, query=query)

    # Format as a dialogue for instruction-tuned models
    dialogue_template = [{"role": "user", "content": formatted_prompt}]
    prompt = tokenizer.apply_chat_template(
        conversation=dialogue_template, tokenize=False, add_generation_prompt=True
    )
    return prompt


# -------------------
# LLM Setup and Generation with TinyLlama
# -------------------
# Use TinyLlama as a smaller, efficient model
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"[INFO] Using model_id: {model_id}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.float16,  # Use float16 for efficiency
    low_cpu_mem_usage=True,  # Optimize memory usage on CPU
)
llm_model.to(device)  # Move model to selected device (CPU or GPU)


# -------------------
# Main RAG Function
# -------------------
def ask(query: str, max_new_tokens: int = 256) -> str:
    """
    Implements RAG: Retrieves relevant context, augments the prompt, and generates an answer using TinyLlama.

    Args:
        query: The user's query string
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Generated answer string
    """
    # Retrieval: Get top relevant resources
    scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings)

    # Create context items from retrieved indices
    context_items = [pages_and_chunks[i] for i in indices]

    # Augmentation: Format prompt with context
    prompt = prompt_formatter(query=query, context_items=context_items)

    # Generation: Tokenize and generate answer
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(
        **input_ids,
        temperature=0.7,  # Controls randomness: lower = deterministic, higher = creative
        do_sample=True,  # Enables sampling for varied outputs
        max_new_tokens=max_new_tokens,
    )

    # Decode generated tokens to text and clean up
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = output_text.replace(prompt, "").strip()  # Remove prompt from output
    return answer


# -------------------
# Example Usage
# -------------------
# List of example queries
query_list = [
    "What are the macronutrients, and what roles do they play in the human body?",
    "What are symptoms of pellagra?",
    "How does saliva help with digestion?",
]

# Select a random query and generate an answer
query = random.choice(query_list)
print(f"\n[INFO] Selected query: {query}")

# Get RAG answer
answer = ask(query=query)
print(f"\n[INFO] RAG Answer:")
print_wrapped(answer)

# Measure and print total execution time
end_time = timer()
total_time = end_time - start_time
print(f"\n[INFO] Total execution time: {total_time:.5f} seconds")
