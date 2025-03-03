import random
import os
import pandas as pd
import torch
import numpy as np
from time import perf_counter as timer
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import textwrap
from dotenv import load_dotenv


# Start timing the entire process
start_time = timer()

# Set device (use GPU with 2GB VRAM if available, fallback to CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# Data Loading
# -------------------
# Load precomputed embeddings and text chunks from CSV
text_chunks_and_embedding_df = pd.read_csv("./text_chunks_and_embeddings_df.csv")

# Convert embedding column from string to numpy array
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df[
    "embedding"
].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

# Convert DataFrame to list of dictionaries
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

# Convert embeddings to PyTorch tensor and move to CPU (for retrieval)
embeddings = torch.tensor(
    np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32
).to(
    "cpu"
)  # Use CPU for embeddings to save GPU memory

# Load sentence transformer model for embedding queries (on CPU)
embedding_model = SentenceTransformer(
    model_name_or_path="all-mpnet-base-v2", device="cpu"
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
    Retrieves top-k relevant resources for a query using cosine similarity.

    Args:
        query: User's query string
        embeddings: Precomputed embeddings of text chunks
        model: Sentence transformer model for embedding the query
        n_resources_to_return: Number of top resources to return

    Returns:
        Tuple of (scores, indices) where scores are similarity scores and indices are positions in pages_and_chunks
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]  # # pyright: ignore
    end_time = timer()
    print(f"[INFO] Retrieval time: {end_time - start_time:.5f} seconds")
    scores, indices = torch.topk(dot_scores, k=n_resources_to_return)
    return scores, indices


# -------------------
# Prompt Formatting
# -------------------
def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """
    Formats a prompt for TinyLlama-Chat model using the correct chat template with system, user, and assistant tags.

    Args:
        query: User's query string
        context_items: List of dictionaries with relevant text chunks

    Returns:
        Formatted prompt string for LLM input
    """
    context = "\n".join([f"- {item['sentence_chunk']}" for item in context_items])

    system_message = "You are a helpful assistant that provides accurate information based on the given context."

    user_message = f"""I need information about the following query:

{query}

Here is the relevant context to help you answer:
{context}

Please provide a detailed response based only on the context provided."""

    # Format using TinyLlama chat template with special tokens
    formatted_prompt = f"<|system|>\n{{{system_message}}}</s>\n<|user|>\n{{{user_message}}}</s>\n<|assistant|>"

    return formatted_prompt


# -------------------
# LLM Setup with TinyLlama GGUF
# -------------------
# Define model details for downloading GGUF file
model_repo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
model_filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
local_dir = "./model/"

# Get Hugging Face token from environment
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print(f"[ERROR] HF_TOKEN not found in .env file")
    print(f"Please create a .env with: HF_TOKEN=your_huggingface_toke")
    exit(1)

# Create model directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)
# Check if the file already exists
expected_path = os.path.join(local_dir, model_filename)
if os.path.exists(expected_path):
    print(f"Model already exists at: {expected_path}")
    model_path = expected_path
else:
    print(f"Downloading model {model_filename} from {model_repo}...")
    model_path = hf_hub_download(
        repo_id=model_repo,
        filename=model_filename,
        local_dir=local_dir,
        token=hf_token,
    )
    print(f"Model downloaded to: {model_path}")

print(f"[INFO] Using model_id: {model_path}")

# Load the quantized model with llama-cpp-python
llm_model = Llama(
    model_path=model_path,
    n_gpu_layers=(
        10 if device == "cuda" else 0
    ),  # Offload 10 layers to GPU if available (fits 2GB VRAM)
    n_ctx=2048,  # Context length, adjust based on VRAM needs
    n_threads=4,  # The number of CPU threads to use
    verbose=False,  # Reduce logging for cleaner output
)


# -------------------
# Main RAG Function
# -------------------


def ask(query: str, max_new_tokens: int = 512) -> str:
    """
    Implements RAG: Retrieves context, augments prompt, and generates an answer with TinyLlama GGUF.

    Args:
        query: User's query string
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

    # Generation: Generate answer with llama-cpp-python (non-streaming)
    output = llm_model(
        prompt,
        max_tokens=max_new_tokens,
        temperature=0.7,  # Controls randomness
        stop=["</s>"],  # Stop token to clean output
        echo=False,  # Ensure no prompt echo in output
    )

    # Extract and clean the answer
    answer = output["choices"][0]["text"].strip()  # # pyright: ignore
    return answer


# -------------------
# Example Usage
# -------------------
# List of example queries
# Create a list of queries
gpt4_questions = [
    "What are the macronutrients, and what roles do they play in the human body?",
    "How do vitamins and minerals differ in their roles and importance for health?",
    "Describe the process of digestion and absorption of nutrients in the human body.",
    "What role does fibre play in digestion? Name five fibre containing foods.",
    "Explain the concept of energy balance and its importance in weight management.",
]

# Manually created question list
manual_questions = [
    "How often should infants be breastfed?",
    "What are symptoms of pellagra?",
    "How does saliva help with digestion?",
    "What is the RDI for protein per day?",
    "water soluble vitamins",
]

query_list = gpt4_questions + manual_questions

# Select a random query and generate an answer
query = random.choice(query_list)
print(f"\n[INFO] Selected query: {query}")

# Retrieval: Get top relevant resources
# scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings)

# Create context items from retrieved indices
# context_items = [pages_and_chunks[i] for i in indices]

# Augmentation: Format prompt with context
# prompt = prompt_formatter(query=query, context_items=context_items)

# PERF: debug point
# print(prompt)

# Simple inference example
# output = llm_model(
#     # "<|system|>\n{You are a story writing assistant.}</s>\n<|user|>\n{Write a story about llamas.}</s>\n<|assistant|>",  # Prompt
#     prompt,
#     max_tokens=512,  # Generate up to 512 tokens
#     stop=[
#         "</s>"
#     ],  # Example stop token - not necessarily correct for this specific model! Please check before using.
#     echo=True,  # Whether to echo the prompt
# )
# Print the output
# print(output["choices"][0]["text"])


# Get RAG answer
answer = ask(query=query)

print(f"\n[INFO] RAG Answer:")
print_wrapped(answer)
#
llm_model._sampler.close()  # # pyright: ignore
llm_model.close()
# Measure and print total execution time
end_time = timer()
total_time = end_time - start_time
print(f"\n[INFO] Total execution time: {total_time:.5f} seconds")
