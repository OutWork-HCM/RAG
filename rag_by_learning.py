import random
import os
import requests
import fitz  # pymupdf, this is better than pypdf, requires pip install pymupdf
from tqdm import tqdm_notebook
from tqdm.auto import tqdm  # for progress bars, requires pip install tqdm
from spacy.lang.en import English  # see https://spacy.io/usage
import pandas as pd
import re
from sentence_transformers import util, SentenceTransformer
import torch
import numpy as np
import textwrap
from time import perf_counter as timer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

start_time = timer()  # Record start time

# Check local GPU capacity
gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
gpu_memory_gb = round(gpu_memory_bytes / (2**30))

if gpu_memory_gb <= 5:
    device = "cpu"  # VRAM is so low, should use "cpu" istead
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# import our textbook
text_chunks_and_embedding_df = pd.read_csv("./text_chunks_and_embeddings_df.csv")


# Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df[
    "embedding"
].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

# Convert texts and embedding df to list of dicts
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

# Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
embeddings = torch.tensor(
    np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32
).to(device)

# PERF: debug pont
# print(embeddings.shape)
# print(text_chunks_and_embedding_df.head())

# Load LLM model
embedding_model = SentenceTransformer(
    model_name_or_path="all-mpnet-base-v2", device=device
)


# Define a helper function to print wrapped text ~ length=80 chars
def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


# Step 1 - FUNCTION OUR SEMANTIC SEARCH PIPELINE
def retrieve_relevant_resource(
    query: str,
    embeddings: torch.Tensor,
    model: SentenceTransformer = embedding_model,
    n_resources_to_return: int = 5,
    print_time: bool = True,
):
    """
    Embeds a query with model and return top k scores and indices from embeddings.
    """

    # embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # dot product query on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]  # pyright: ignore
    end_time = timer()

    if print_time:
        print(
            f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds."
        )
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

    return scores, indices


def print_top_results_and_scores(
    query: str,
    embeddings: torch.Tensor,
    pages_and_chunks: list[dict] = pages_and_chunks,
    n_resources_to_return: int = 5,
):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.
    """
    scores, indices = retrieve_relevant_resource(
        query=query, embeddings=embeddings, n_resources_to_return=n_resources_to_return
    )
    print(f"Query: {query}\n")
    print("Results: ")
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print_wrapped(pages_and_chunks[index]["sentence_chunk"])
        print(f"Page number: {pages_and_chunks[index]['page_number']}")
        print("\n")


# PERF: debug point
"""
query = " symptoms of pellagra"
# Get just teh scores and indices of top related results
scores, indices = retrieve_relevant_resource(query=query, embeddings=embeddings)
print(scores, indices)
# Print out the texts of the top scores
print_top_results_and_scores(query=query, embeddings=embeddings)
"""

# GETTING AN LLM FOR LOCAL GENERATION
# To find open-source LLMs, ref to Hugging Face open LLM leaderboard (https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/)
# Or want to find a quantized modes, please ref to TheBloke on Hugging Face (https://huggingface.co/TheBloke)

# print(f"Your GPU mem is: {gpu_memory_gb}") # PERF: debug point
if gpu_memory_gb < 5.1:
    print(
        f"Your available GPU memory is {gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization."
    )
    use_quantization_config = False
    model_id = "google/gemma-2b-it"
elif gpu_memory_gb < 8.1:
    print(
        f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision."
    )
    use_quantization_config = True
    model_id = "google/gemma-2b-it"
elif gpu_memory_gb < 19.0:
    print(
        f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision."
    )
    use_quantization_config = False
    model_id = "google/gemma-2b-it"
else:  # gpu_memory_gb > 19.0:
    print(
        f"GPU memory: {gpu_memory_gb} | Recommend model: Gemma 7B in 4-bit or float16 precision."
    )
    use_quantization_config = False
    model_id = "google/gemma-7b-it"

# PERF: debug point
# print(f"use_quantization_config set to: {use_quantization_config}")
# print(f"model_id set to: {model_id}")

# Create quantization config for smaller model loading (optional)
# Requires !pip install bitsandbytes accelerate, see: https://github.com/TimDettmers/bitsandbytes, https://huggingface.co/docs/accelerate/
# For models that require 4-bit quantization (use this if you have low GPU memory available)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Dinh nghia device_map
# custom_device_map = {"model.decoder": "cuda", "model.lm_head": "cuda", "": "cpu"}

# Bonus: Setup Flash Attention 2 for faster inference, default to "sdpa" or "scaled dot product attention" if it's not available
# Flash Attention 2 requires NVIDIA GPU compute capability of 8.0 or above, see: https://developer.nvidia.com/cuda-gpus
# Requires !pip install flash-attn, see: https://github.com/Dao-AILab/flash-attention
if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"

print(f"[INFO] Using attention implementation: {attn_implementation}")
print(f"[INFO] Using model_id: {model_id}")

# Instantiate tokenizers
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

# Instantiate the model
llm_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.float16,
    quantization_config=quantization_config if use_quantization_config else None,
    low_cpu_mem_usage=True,  # Giam tai bo nho CPU khi load mo hinh
    attn_implementation=attn_implementation,
)
if (
    not use_quantization_config
):  # quantization takes care of device setting automatically, so if it's not used, send model to GPU
    llm_model.to(device)

llm_model.to(device)
# print(llm_model)


# Get number of parameters of the model
def get_model_num_params(model: torch.nn.Module):
    return sum([param.numel() for param in model.parameters()])


print(get_model_num_params(llm_model))


# Get the models memory requirements
def get_model_mem_size(model: torch.nn.Module):
    """
    Get how much memory a PyTorch model takes up.

    See: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
    """
    # Get model parameters and buffer sizes
    mem_params = sum(
        [param.nelement() * param.element_size() for param in model.parameters()]
    )
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Calculate various model sizes
    model_mem_bytes = mem_params + mem_buffers  # in bytes
    model_mem_mb = model_mem_bytes / (1024**2)  # in megabytes
    model_mem_gb = model_mem_bytes / (1024**3)  # in gigabytes

    return {
        "model_mem_bytes": model_mem_bytes,
        "model_mem_mb": round(model_mem_mb, 2),
        "model_mem_gb": round(model_mem_gb, 2),
    }


# NOTE: GENERATING TEXT WITH OUR LLM
input_text = (
    f"What are the macronutrients, and what roles do they play in the human body?"
)

# Create prompt template for instruction-tuned model
dialogue_template = [{"role": "user", "content": input_text}]
# Apply the chat template
prompt = tokenizer.apply_chat_template(
    conversation=dialogue_template, tokenize=False, add_generation_prompt=True
)

# Tokenize the input text (turn it into numbers/tensor) and send it to GPU
input_ids = tokenizer(prompt, return_tensors="pt").to(device)
#
# # Generate outputs passed on the tokenized input
# outputs = llm_model.generate(
#     **input_ids, max_new_tokens=256
# )  # Define the maximum of new tokens to create
#
# # Decode teh output tokens to text
# outputs_decoded = tokenizer.decode(outputs[0])
# print(f"Model Output: \n {outputs_decoded}\n")

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

# NOTE: Retrieval Process (R in RAG)
query = random.choice(query_list)
print(f"User question: {query}")

# Get scores and indices of top related results
scores, indices = retrieve_relevant_resource(query=query, embeddings=embeddings)

# Create a list of context items
context_items = [pages_and_chunks[i] for i in indices]

# print(context_items[0])
# print(len(context_items))


# NOTE: Augmenting our prompt with context items (A in RAG)
def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
\nExample 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
\nExample 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Update base prompt with context items and query
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [{"role": "user", "content": base_prompt}]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(
        conversation=dialogue_template, tokenize=False, add_generation_prompt=True
    )
    return prompt


# Format prompt with context items
prompt = prompt_formatter(query=query, context_items=context_items)
print(prompt)

input_ids = tokenizer(prompt, return_tensors="pt").to(device)

# NOTE: Generate an output of tokens (G in RAG)
# Generate an output of tokens
outputs = llm_model.generate(
    **input_ids,
    temperature=0.7,  # lower temperature = more deterministic outputs, higher temperature = more creative outputs
    do_sample=True,  # whether or not to use sampling, see https://huyenchip.com/2024/01/16/sampling.html for more
    max_new_tokens=256,
)  # how many new tokens to generate from prompt

# Turn the output tokens into text
output_text = tokenizer.decode(outputs[0])

print(f"Query: {query}")
print(f"RAG answer:\n{output_text.replace(prompt, '')}")


# Define a nice function to return answer
def ask(
    query,
    temperature=0.7,
    max_new_tokens=512,
    format_answer_text=True,
    return_answer_only=True,
):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """

    # Get just the scores and indices of top related results
    scores, indices = retrieve_relevant_resource(query=query, embeddings=embeddings)

    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]

    # Add score to context item
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu()  # return score back to CPU

    # Format the prompt with context items
    prompt = prompt_formatter(query=query, context_items=context_items)

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate an output of tokens
    outputs = llm_model.generate(
        **input_ids,
        temperature=temperature,
        do_sample=True,
        max_new_tokens=max_new_tokens,
    )

    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        # Replace special tokens and unnecessary help message
        output_text = (
            output_text.replace(prompt, "")
            .replace("<bos>", "")
            .replace("<eos>", "")
            .replace("Sure, here is the answer to the user query:\n\n", "")
        )

    # Only return the answer without the context items
    if return_answer_only:
        return output_text

    return output_text, context_items


# Answer query with context and return context
answer, context_items = ask(
    query=query, temperature=0.7, max_new_tokens=512, return_answer_only=True
)

print(f"Answer:\n")
print_wrapped(answer)

end_time = timer()
total_time = end_time - start_time
print(f"[INFO] Total executeion time: {total_time:.5f} seconds")
