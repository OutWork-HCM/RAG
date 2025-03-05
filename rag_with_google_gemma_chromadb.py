import random
import os
import pandas as pd
import torch
import numpy as np
from time import perf_counter as timer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import textwrap
import chromadb

# Start timing the entire process
start_time = timer()

# Set device to CPU if GPU memory is low, otherwise use GPU if available
gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
gpu_memory_gb = round(gpu_memory_bytes / (2**30))
device = "cpu" if gpu_memory_gb <= 5 else "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA if running on CPU

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

    # Base prompt with instructions and examples for the LLM
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
# LLM Setup and Generation
# -------------------
# Set default model and load tokenizer/model based on GPU capacity
model_id = "google/gemma-2b-it"
use_quantization_config = False if gpu_memory_gb < 5.1 else True

print(f"[INFO] Using model_id: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the LLM with appropriate configuration
llm_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.float16,
    quantization_config=(
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        if use_quantization_config
        else None
    ),
    low_cpu_mem_usage=True,
    device_map="auto" if use_quantization_config else None,
)
if not use_quantization_config:
    llm_model.to(device)


# -------------------
# Main RAG Function
# -------------------
def ask(query: str, max_new_tokens: int = 256) -> str:
    """
    Implements RAG: Retrieves relevant context, augments the prompt, and generates an answer.

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
