import os
import random
import torch
import numpy as np
import faiss
import chromadb
import textwrap
from time import perf_counter as timer
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Start timing the entire process
start_time = timer()

# Load Hugging Face token từ .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("[ERROR] HF_TOKEN not found in .env file")
    print("Please create a .env file with: HF_TOKEN=your_huggingface_token")
    exit(1)

# ------------------------------
# Cấu hình thiết bị (GPU/CPU)
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(f"[INFO] Using device: {device}")

# ------------------------------
# Load dữ liệu từ ChromaDB
# ------------------------------
db_dir = "./myDB"
db_name = "verilog_text"
client = chromadb.PersistentClient(path=db_dir)
collection = client.get_or_create_collection(name=db_name)

if os.path.exists("cached_embeddings.npy"):
    embeddings = torch.tensor(np.load("cached_embeddings.npy"), dtype=torch.float32).to(
        "cpu"
    )
    with open("cached_metadata.npy", "rb") as f:
        pages_and_chunks = np.load(f, allow_pickle=True)
else:
    data = collection.get(include=["metadatas", "embeddings"], limit=10000)
    pages_and_chunks = [
        {
            "sentence_chunk": item["text"],
            "page_number": item["page_number"],
            "filename": item["filename"],
        }
        for item in data["metadatas"]
    ]
    embeddings = torch.tensor(np.array(data["embeddings"]), dtype=torch.float32).to(
        "cpu"
    )
    np.save("cached_embeddings.npy", np.array(data["embeddings"]))
    np.save("cached_metadata.npy", np.array(pages_and_chunks, dtype=object))

# ------------------------------
# FAISS Index cho tìm kiếm nhanh
# ------------------------------
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embeddings.numpy())

# ------------------------------
# Load SentenceTransformer
# ------------------------------
embedding_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")


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


# ------------------------------
# Định dạng prompt
# ------------------------------
def prompt_formatter(query: str, context_items: list[dict]) -> str:
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    base_prompt = """Based on the following context items, please answer the query.
Make sure your answers are explanatory and based on the provided context.

Context:
{context}

User query: {query}
Answer:"""
    return base_prompt.format(context=context, query=query)


# ------------------------------
# Tải mô hình LLM về local
# ------------------------------
local_dir = "./model"
model_id = "google/gemma-2b-it"

if not os.path.exists(local_dir):
    os.makedirs(local_dir)

print(f"[INFO] Downloading model {model_id} to {local_dir}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, cache_dir=local_dir)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    cache_dir=local_dir,
    device_map=device if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    # quantization_config=BitsAndBytesConfig(
    #     load_in_8bit=True if device == "cuda" else False,
    #     bnb_8bit_compute_dtype=torch.float16
    # )
)

if device == "cuda":
    llm_model = torch.compile(llm_model)

print("[INFO] Model loaded successfully!")


# ------------------------------
# Hàm tìm kiếm tài liệu với FAISS
# ------------------------------
def retrieve_relevant_resources(query: str, n_resources_to_return: int = 5):
    query_embedding = embedding_model.encode(query, convert_to_tensor=False).reshape(
        1, -1
    )
    scores, indices = faiss_index.search(query_embedding, k=n_resources_to_return)
    return scores[0], indices[0]


# ------------------------------
# Hàm chính cho RAG
# ------------------------------
def ask(query: str, max_new_tokens: int = 256) -> str:
    scores, indices = retrieve_relevant_resources(query=query)
    context_items = [pages_and_chunks[i] for i in indices]
    prompt = prompt_formatter(query, context_items)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output_tokens = llm_model.generate(
        input_ids,
        temperature=0.7,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        # stream=True
    )

    output_text = ""
    for token in output_tokens:
        text_piece = tokenizer.decode(token, skip_special_tokens=True)
        # print(text_piece, end="", flush=True)
        output_text += text_piece

    return output_text.strip()


# ------------------------------
# Chạy thử nghiệm với danh sách truy vấn
# ------------------------------
query_list = [
    "What are the four levels of abstraction in Verilog, and how do they differ?",
    "Explain the difference between blocking and non-blocking assignments in Verilog.",
    "What is the purpose of an always block in Verilog, and how does it differ from an initial block?",
    "Describe the differences between procedural and continuous assignments in Verilog.",
    "How does a finite state machine (FSM) work in Verilog, and what are its key components?",
    "What are the different types of modeling styles available in Verilog, and when should each be used?",
    "Explain how module instantiation works in Verilog and how it helps in hierarchical design.",
    "What is a testbench in Verilog, and why is it important for verifying digital designs?",
    "How do you use the generate statement in Verilog, and in what scenarios is it useful?",
    "What are system tasks and functions in Verilog, and provide examples of their usage?",
]
query = random.choice(query_list)
print(f"\n[INFO] Selected query: {query}")

answer = ask(query=query)
if "Answer:" in answer:
    answer = answer.split("Answer:")[1].strip()

print(f"\n[INFO] RAG Answer:")
print_wrapped(answer)

# ------------------------------
# Tính toán thời gian chạy
# ------------------------------
end_time = timer()
total_time = end_time - start_time
print(f"\n[INFO] Total execution time: {total_time:.5f} seconds")
