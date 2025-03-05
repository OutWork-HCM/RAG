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
import chromadb

# Start timing the entire process
start_time = timer()

# Set device (use GPU with 2GB VRAM if available, fallback to CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Data Loading from chromadb (persistent database)
# -------------------------------------------------
db_dir = "./myDB"
db_name = "verilog_text"
client = chromadb.PersistentClient(path=db_dir)  # pyright: ignore
collection = client.get_or_create_collection(name=db_name)
data = collection.get(include=["metadatas", "embeddings"])

# Load precomputed embeddings and text chunks
# pages_and_chunks = data["metadatas"]
pages_and_chunks = [
    {
        "sentence_chunk": item["text"],
        "page_number": item["page_number"],
        "filename": item["filename"],
    }
    for item in data["metadatas"]
]

all_embeddings = data["embeddings"]


# Convert embeddings to PyTorch tensor and move to CPU (for retrieval)
embeddings = torch.tensor(np.array(all_embeddings), dtype=torch.float32).to(
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
    "What is Verilog HDL, and how does it differ from VHDL?",
    "What are the different levels of abstraction in Verilog?",
    "What are the main applications of Verilog in digital design?",
    "How does Verilog handle concurrency?",
    "What are the advantages and disadvantages of using Verilog?",
    "What are the basic data types in Verilog?",
    "What is the difference between wire and reg in Verilog?",
    "How do you declare and initialize variables in Verilog?",
    "What are the different types of operators available in Verilog?",
    "How does Verilog handle signed and unsigned numbers?",
    "What is the difference between structural, behavioral, and dataflow modeling in Verilog?",
    "How do you implement a multiplexer using structural modeling in Verilog?",
    "What is the purpose of an initial block in Verilog?",
    "How does the always block work, and how is it different from an initial block?",
    "How do you create a finite state machine (FSM) in Verilog?",
    "What is the difference between blocking and non-blocking assignments in Verilog?",
    "How do you use if-else and case statements in Verilog?",
    "What are the different types of loops available in Verilog?",
    "What is the purpose of disable and fork-join statements in Verilog?",
    "How can you avoid race conditions in Verilog designs?",
    "What is a testbench in Verilog, and why is it important?",
    "How do you use $monitor, $display, and $strobe in a testbench?",
    "What is the purpose of $dumpfile and $dumpvars in Verilog simulation?",
    "How do you generate random test vectors in Verilog?",
    "What are some common debugging techniques used in Verilog?",
    "What is the difference between synthesizable and non-synthesizable Verilog code?",
    "What constructs should be avoided for synthesis in Verilog?",
    "How do you implement a clock divider in Verilog?",
    "What are the differences between always @* and always @(posedge clk)?",
    "How do synthesis tools optimize Verilog code?",
    "What is SystemVerilog, and how does it extend Verilog?",
    "How do you implement pipelining in Verilog?",
    "What are parameterized modules, and how do you use them?",
    "How do you model asynchronous circuits in Verilog?",
    "What is a race condition, and how can it be prevented in Verilog?",
]

# Manually created question list
manual_questions = []
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
answer = ask(query=query, max_new_tokens=2048)

print(f"\n[INFO] RAG Answer:")
print_wrapped(answer)
#
llm_model._sampler.close()  # # pyright: ignore
llm_model.close()
# Measure and print total execution time
end_time = timer()
total_time = end_time - start_time
print(f"\n[INFO] Total execution time: {total_time:.5f} seconds")
