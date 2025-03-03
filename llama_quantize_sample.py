from llama_cpp import Llama
from time import perf_counter as timer
import torch

# Start timing the entire process
start_time = timer()

# Set device (use GPU with 2GB VRAM if available, fallback to CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model = Llama(
    model_path="./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",  # Download the model file first
    n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=4,  # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=35,  # The number of layers to offload to GPU, if you have GPU acceleration available
)

# Simple inference example
output = model(
    "<|system|>\n{You are a story writing assistant.}</s>\n<|user|>\n{Write a story about llamas.}</s>\n<|assistant|>",  # Prompt
    max_tokens=512,  # Generate up to 512 tokens
    stop=[
        "</s>"
    ],  # Example stop token - not necessarily correct for this specific model! Please check before using.
    echo=True,  # Whether to echo the prompt
)
# Print the output
print(output["choices"][0]["text"])

# model = Llama(
#     model_path="./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", n_gpu_layers=10, verbose=False
# )

# generation_kwargs = {"max_tokens": 2048}
#
# messages = [
#     {"role": "system", "content": "Be a helpful assistant"},
#     {"role": "user", "content": "Tell me a story about American"},
# ]
#
# res = model.create_chat_completion(messages=messages, **generation_kwargs)
#
# print(res["choices"][0])

model._sampler.close()
model.close()

# Measure and print total execution time
end_time = timer()
total_time = end_time - start_time
print(f"\n[INFO] Total execution time: {total_time:.5f} seconds")
