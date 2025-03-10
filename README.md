# Chat with Multiple PDFs

A Streamlit-based application that allows users to upload PDF documents and chat with them. The app leverages LangChain, FAISS, and vector embeddings to process, chunk, and retrieve relevant information from PDFs, enabling a conversational interface powered by a language model.

## Features

- **PDF Upload**: Easily upload one or more PDFs.
- **Text Extraction**: Extract text from PDFs using PyPDF2.
- **Document Chunking**: Split the extracted text into manageable chunks with configurable size and overlap.
- **Embeddings & Retrieval**: Generate embeddings (using HuggingFaceInstructEmbeddings) and store them in a FAISS vector store for efficient retrieval.
- **Conversational Interface**: Engage in a chat where user queries are matched with relevant text chunks from the documents using a Conversational Retrieval Chain.
- **Streamlit UI**: Interactive and user-friendly interface built with Streamlit.

## Requirements

- Python 3.9 or higher
- [Streamlit](https://streamlit.io/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [LangChain](https://github.com/hwchase17/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Torch](https://pytorch.org/)
- [HuggingFaceHub](https://huggingface.co/docs/huggingface_hub)

## Installation

1. **Clone the repository:**
   ```bash
   git clone -b feature/rag_langchain git@github.com:OutWork-HCM/RAG.git
   cd RAG
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root (if needed) to store any API keys or configuration settings.

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Upload and Process PDFs:**
   - Use the sidebar to upload your PDF files.
   - Click the **Process** button to extract text, split it into chunks, and build the vector store.

3. **Chat with Your Documents:**
   - Enter a question in the text input field at the top of the app.
   - The app retrieves relevant chunks from your PDFs and generates an answer using the conversational chain.

## Code Overview

- **`get_pdf_text(pdf_docs)`**: Reads and extracts text from the uploaded PDF documents.
- **`get_text_chunks(raw_text)`**: Splits the extracted text into overlapping chunks using a character-based splitter.
- **`get_vectorstore_withOpenAI(chunks)`** and **`get_vectorstore_withOpenInstructor(chunks)`**: Generate vector stores from text chunks using different embedding models.
- **`get_conversation_chain(vectorstore)`**: Creates a conversational retrieval chain that connects the vector store with a language model (Flan-T5-XXL) and maintains conversation history.
- **`handle_userinput(user_question)`**: Processes user input, retrieves relevant information from the vector store, and displays the conversation.
- **`main()`**: Orchestrates the application, including PDF upload, text processing, vector store creation, and user interface setup.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or feature suggestions.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Thanks to the developers of Streamlit, LangChain, FAISS, and other libraries used in this project.
- Inspired by modern approaches to Retrieval-Augmented Generation (RAG) in AI applications.

---

Feel free to modify or expand this README to better suit your project's specific details and requirements.
