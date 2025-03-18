# # RAG (Retrieval-Augmented Generation) System

This project implements a Retrieval-Augmented Generation (RAG) system using Python. It allows you to load PDF documents, split them into chunks, embed the chunks, store them in a vector database, and query the database using a large language model (LLM) for generating responses.
![Alt text](./RAGwithOpenSource.png)
## Project Structure

- **populate_database.py**: Script to load PDFs, split them into chunks, and store them in ChromaDB.
- **query_data.py**: Script to query the ChromaDB and generate responses using the Gemma 3 model.
- **requirements.txt**: List of all required Python packages.
- **data/**: Directory containing PDF files to be loaded. 
- **myDB/**: Directory where ChromaDB stores its data.
## Notes

- Ensure that all dependencies are installed by running:
    ```bash
	pip install -r requirements.txt  
	```

- The `data/` directory should contain the PDF files you want to process.

- The `myDB/` directory is used by ChromaDB to store the vector database. You can reset it using the `--reset` flag in `populate_database.py`.
## Requirements

- GPU VRAM > 4G for "google/gemma-3-1b-it" and > 10G for "google/gemma-3-4b-it"
- CUDA 12.4
- Python 3.10.10
