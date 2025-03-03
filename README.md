# Nutrition Q&A with RAG Pipelines

This project processes a nutrition PDF and answers questions using Retrieval-Augmented Generation (RAG) pipelines. It includes two RAG options:
1. **TinyLlama Pipeline**: Lightweight, quantized model for low-resource systems.
2. **Google Gemma Pipeline**: More powerful model for systems with decent GPU/CPU.

## What It Does

* Downloads a nutrition PDF and turns its text into searchable embeddings (`doc_processing_and_embedding.py`).
* Answers questions (e.g., "What are macronutrients?") using:
   * `rag_pipeline.py` (TinyLlama).
   * `rag_with_google_gemma.py` (Google Gemma).

## Prerequisites

* **System**: Best on Linux (bash script provided). Windows/Mac may need tweaks.
* **Hardware**:
   * TinyLlama: Runs on CPU or GPU (2GB+ VRAM).
   * Gemma: Better with GPU (5GB+ VRAM), falls back to CPU if low memory.
* **Tools**: Conda and a Hugging Face token.
* **Space**: ~2GB for models and files.

## Quick Setup

### 1. Clone the Repo
```bash
git clone git@github.com:OutWork-HCM/RAG.git
cd RAG
```

### 2. Add Your Hugging Face Token
Create a .env file:
```text
HF_TOKEN=your_huggingface_token_here
```
Get your token from [Hugging Face](https://huggingface.co/settings/tokens).

### 3. Set Up the Environment
Run the setup script:
```bash
chmod +x setup_env.sh
./setup_env.sh RAG
```
* Replace `myenv` with your preferred name.
* Installs Python 3.11, PyTorch, Sentence Transformers, and more.

### 4. Activate the Environment
```bash
conda activate RAG
```

## Usage

### Step 1: Process the PDF
Run the document processor to create embeddings:
```bash
python doc_processing_and_embedding.py
```
* Downloads `human-nutrition-text.pdf`.
* Saves embeddings to `text_chunks_and_embeddings_df.csv`.

### Step 2: Ask Questions
Choose a pipeline:

#### Option 1: TinyLlama (Lightweight)
```bash
python rag_with_Llama_quantize.py
```
* Uses TinyLlama-1.1B-Chat (700MB model).
* Good for low-memory systems.

#### Option 2: Google Gemma (Powerful)
```bash
python rag_with_google_gemma.py
```
* Uses google/gemma-2b-it (larger model).
* Better for systems with 5GB+ GPU memory or decent CPU.

## Example Output
```text
[INFO] Selected query: What are macronutrients?
[INFO] Retrieval time: 0.00050 seconds
[INFO] RAG Answer: Macronutrients are proteins, fats, and carbohydrates that provide energy...
[INFO] Total execution time: 12.34567 seconds
```

## Customize It

* **Add Questions**: Edit `query_list` in either RAG script.
* **Answer Length**: Change `max_new_tokens` in `ask()`.
* **Context Size**: Adjust `n_resources_to_return` in `retrieve_relevant_resources()`.

## Files

* **`doc_processing_and_embedding.py`**: Processes PDF and creates `text_chunks_and_embeddings_df.csv`.
* **`rag_with_Llama_quantize.py`**: TinyLlama-based RAG pipeline.
* **`rag_with_google_gemma.py`**: Google Gemma-based RAG pipeline.
* **`setup_env.sh`**: Sets up the environment.
* **`.env`**: Stores your Hugging Face token.

## Dependencies

* Python 3.11
* PyTorch (CUDA 12.4 for GPU)
* Sentence Transformers
* Llama-cpp-python (TinyLlama)
* Transformers & BitsAndBytes (Gemma)
* Pandas, NumPy, SpaCy, etc. (see `setup_env.sh`)

## Tips

* **TinyLlama**: Fast on CPU, downloads a small model.
* **Gemma**: Needs more memory, uses quantization if GPU < 5GB.
* **No GPU?** Both work on CPU, but Gemma may be slow.

## Troubleshooting

* **Token Error**: Check `.env` has a valid `HF_TOKEN`.
* **CSV Missing**: Run `doc_processing_and_embedding.py` first.
* **Memory Issues**: Use TinyLlama or ensure 5GB+ GPU for Gemma.

## Contributing

Ideas? Open an issue or send a pull request!

## License

MIT License
