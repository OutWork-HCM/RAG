import streamlit as st
import os
import shutil
from get_embedding import get_embedding
import PyPDF2
from PyPDF2.generic import TextStringObject
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from langchain.schema.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from tqdm import tqdm
from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import time
from chromadb import PersistentClient
from chromadb.errors import ChromaError

# --- Thư mục và hàm xử lý của bạn ---
CHROMA_PATH = "./myDB" # Folder for ChromaDB
DATA_PATH_EN = "./data/en" # English PDFs
DATA_PATH_VI = "./data/vi" # Vietnamese PDFs

def ensure_directories_exist():
    for path in [CHROMA_PATH, DATA_PATH_EN, DATA_PATH_VI]:
        if not os.path.exists(path):
            os.makedirs(path)

# def clear_database():
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)
#         st.sidebar.success(f"Deleted existing database at {CHROMA_PATH}")
def clear_database():
    try:
        # Khởi tạo client Chroma trực tiếp
        client = PersistentClient(path=CHROMA_PATH)
        # Thử xóa collection RAGDB
        client.delete_collection(name="RAGDB")
        st.sidebar.success("✅ Đã xóa thành công collection RAGDB")
    except ValueError as e:
        if "does not exist" in str(e):
            st.sidebar.warning("⚠️ Collection RAGDB không tồn tại")
        else:
            st.sidebar.error(f"❌ Lỗi khi xóa collection: {str(e)}")
    except ChromaError as e:
        st.sidebar.error(f"❌ Lỗi Chroma: {str(e)}")
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi không xác định: {str(e)}")

def get_pdf_metadata(file_path: str) -> dict:
    meta_dict = {}
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            meta = reader.metadata or {}
            total_pages = len(reader.pages)

            def get_meta_value(key, default=""):
                value = meta.get(key, default)
                if isinstance(value, bytes):
                    try:
                        # Decode UTF-16 (thường dùng cho BOM)
                        decoded = value.decode("utf-16").strip("\x00")
                    except UnicodeDecodeError:
                        # Fallback sang Latin-1 nếu có lỗi
                        decoded = value.decode("latin-1", errors="ignore")
                    return decoded
                elif isinstance(value, TextStringObject):
                    return str(value)
                else:
                    return str(value)

            meta_dict = {
                "producer": get_meta_value("/Producer", ""),
                "creator": get_meta_value("/Creator", ""),
                "creationdate": get_meta_value("/CreationDate", ""),
                "source": file_path,
                "file_name": os.path.basename(file_path),
                "total_pages": total_pages,
                "format": get_meta_value("/PDFFormat", "PDF 1.x"),
                "title": get_meta_value("/Title", ""),
                "author": get_meta_value("/Author", ""),
                "subject": get_meta_value("/Subject", ""),
                "keywords": get_meta_value("/Keywords", ""),
                "moddate": get_meta_value("/ModDate", ""),
            }
    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")
    return meta_dict

def split_into_pages(text: str, total_pages: int) -> list:
    # Ensure text is a string before proceeding
    if not isinstance(text, str):
        st.sidebar.warning("split_into_pages received non-string input, returning empty list.")
        return []
    if total_pages <= 0:
        return [text] if text else [] # Return empty list if text is empty

    text_length = len(text)
    if text_length == 0:
        return []

    approx_page_length = text_length // total_pages if total_pages > 0 else text_length
    if approx_page_length == 0: # Avoid issues with very short texts and many pages
         approx_page_length = 1

    pages = []
    start = 0
    for i in range(total_pages):
        if start >= text_length:
            break
        if i == total_pages - 1:
            pages.append(text[start:])
            break

        end = start + approx_page_length
        # Ensure end doesn't exceed text length prematurely
        end = min(end, text_length)

        # Find sentence boundary if possible within reasonable range
        search_limit = min(start + approx_page_length + 100, text_length) # Limit search range
        found_boundary = False
        temp_end = end
        while temp_end < search_limit:
             if text[temp_end] in ".!?":
                 end = temp_end + 1
                 found_boundary = True
                 break
             temp_end += 1

        # If no boundary found, just use approx length (or ensure we don't create empty chunks)
        if not found_boundary and end <= start:
             end = start + 1 # Take at least one character

        # Ensure we don't exceed bounds
        end = min(end, text_length)

        page_text = text[start:end]
        if page_text: # Only add non-empty pages
            pages.append(page_text)
        start = end

    return [p for p in pages if p] # Filter out any potentially empty strings again

def load_vi_documents() -> list[Document]:
    config = {
        "output_format": "markdown",
        "force_ocr": True,
        "languages": "vi,en",
    }
    config_parser = ConfigParser(config)
    model_settings = create_model_dict()
    # Check if model settings are valid
    if not model_settings:
        st.sidebar.error("Failed to create marker model dictionary. Check marker setup.")
        return []
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=model_settings,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )

    documents = []
    pdf_files = [f for f in os.listdir(DATA_PATH_VI) if f.lower().endswith(".pdf")]
    if not pdf_files:
        st.sidebar.warning(f"No Vietnamese PDF files found in {DATA_PATH_VI}")
        return documents

    st.sidebar.write(f"Found {len(pdf_files)} Vietnamese PDF files in {DATA_PATH_VI}")
    for pdf_file in tqdm(pdf_files, desc="Loading Vietnamese PDF files"):
        file_path = os.path.join(DATA_PATH_VI, pdf_file)
        try:
            pdf_meta = get_pdf_metadata(file_path)
            total_pages_meta = pdf_meta.get("total_pages", 1) # From PyPDF2

            # Use marker for conversion
            rendered = converter(file_path)
            if rendered is None:
                 st.sidebar.error(f"❌ Marker failed to convert {pdf_file}")
                 continue

            extracted = text_from_rendered(rendered)
            if not extracted or not isinstance(extracted[0], str):
                 st.sidebar.error(f"❌ Marker failed to extract text from {pdf_file}")
                 continue

            text = extracted[0]
            if not text: # Skip if extracted text is empty
                st.sidebar.warning(f"⚠️ No text extracted from {pdf_file}")
                continue

            # Get page count from marker metadata if available, otherwise use PyPDF2's count
            page_stats = rendered.metadata.get("page_stats", [])
            total_pages_marker = len(page_stats) if page_stats else 0
            total_pages = max(total_pages_meta, total_pages_marker, 1) # Use the max page count, ensure at least 1

            st.sidebar.write(f"✅ Loaded {pdf_file}: ~{total_pages} logical pages")

            # Split the single extracted text blob into approximate pages
            page_texts = split_into_pages(text, total_pages)

            for i, page_text in enumerate(page_texts):
                 if not page_text: continue # Skip empty pages after split
                 page_meta = pdf_meta.copy()
                 page_meta["page"] = i # Page number based on split
                 doc = Document(page_content=page_text, metadata=page_meta)
                 documents.append(doc)

        except Exception as e:
            st.sidebar.error(f"❌ Error loading {pdf_file}: {str(e)}")
            import traceback
            st.sidebar.text(traceback.format_exc()) # Print full traceback for debugging

    st.sidebar.write(f"Total Vietnamese documents created: {len(documents)}")
    return documents


def load_en_documents() -> list[Document]:
    documents = []
    pdf_files = [f for f in os.listdir(DATA_PATH_EN) if f.lower().endswith(".pdf")]
    if not pdf_files:
        st.sidebar.warning(f"No English PDF files found in {DATA_PATH_EN}")
        return documents

    st.sidebar.write(f"Found {len(pdf_files)} English PDF files in {DATA_PATH_EN}")
    for pdf_file in tqdm(pdf_files, desc="Loading English PDF files"):
        file_path = os.path.join(DATA_PATH_EN, pdf_file)
        try:
            loader = PyMuPDFLoader(file_path=file_path)
            docs = loader.load() # PyMuPDFLoader loads page by page
            # Ensure page_content is not None
            valid_docs = [d for d in docs if d.page_content is not None]
            if len(valid_docs) != len(docs):
                 st.sidebar.warning(f"⚠️ Found pages with None content in {pdf_file}")
            documents.extend(valid_docs)
            st.sidebar.write(f"✅ Loaded {pdf_file}: {len(valid_docs)} pages")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {pdf_file}: {str(e)}")
            import traceback
            st.sidebar.text(traceback.format_exc())
    st.sidebar.write(f"Total English pages load: {len(documents)}")
    return documents

def load_all_documents() -> list[Document]:
    docs_en = load_en_documents()
    docs_vi = load_vi_documents()
    # Filter out any documents with None page_content just in case
    all_docs = [doc for doc in (docs_en + docs_vi) if doc.page_content is not None]

    if not all_docs:
         st.sidebar.error(
             "No valid documents found or loaded from PDF files."
         )
         return []
    elif not docs_en:
        st.sidebar.warning(f"No English PDF files loaded from {DATA_PATH_EN}")
    elif not docs_vi:
        st.sidebar.warning(f"No Vietnamese PDF files loaded from {DATA_PATH_VI}")

    return all_docs

def chunk_documents(documents: list[Document]) -> list[Document]:
    if not documents:
        st.sidebar.warning("No documents provided for chunking.")
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False
    )
    st.sidebar.write(f"Splitting {len(documents)} documents into chunks")
    # Ensure we only split documents with actual content
    valid_docs = [doc for doc in documents if doc.page_content]
    if len(valid_docs) < len(documents):
         st.sidebar.warning(f"Skipped {len(documents) - len(valid_docs)} documents with empty content during chunking.")

    if not valid_docs:
         st.sidebar.error("No valid documents with content to chunk.")
         return []

    chunks = text_splitter.split_documents(valid_docs)
    st.sidebar.write(f"Created {len(chunks)} chunks")
    return chunks

def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    page_chunk_counts = {}
    processed_chunks = []
    for chunk in tqdm(chunks, desc="Calculating chunk IDs"):
        # Ensure metadata exists and has needed keys
        if not hasattr(chunk, 'metadata') or chunk.metadata is None:
             st.sidebar.warning("Chunk missing metadata, skipping ID calculation.")
             continue # Skip this chunk

        source = chunk.metadata.get("source", "unknown_source")
        # Handle potential None or missing page number from metadata
        page = chunk.metadata.get("page", "unknown_page") # Use 'unknown_page' if None or missing

        current_page_id = f"{source}:{page}"
        page_chunk_counts[current_page_id] = page_chunk_counts.get(current_page_id, 0)
        chunk_id = f"{current_page_id}:{page_chunk_counts[current_page_id]}"
        page_chunk_counts[current_page_id] += 1

        # Update chunk metadata safely
        chunk.metadata["id"] = chunk_id
        processed_chunks.append(chunk)
    return processed_chunks


def add_to_chroma(chunks: list[Document]):
    if not chunks:
        st.sidebar.warning("No chunks to add to Chroma.")
        return

    embedding_function = get_embedding()
    if embedding_function is None:
        st.sidebar.error("Failed to get embedding function. Cannot add to Chroma.")
        return

    try:
        db = Chroma(
            collection_name="RAGDB",
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function,
        )

        chunks_with_ids = calculate_chunk_ids(chunks)
        # Filter chunks again to ensure they have an ID and content
        valid_chunks = [c for c in chunks_with_ids if c.metadata.get("id") and c.page_content]

        if not valid_chunks:
             st.sidebar.error("No valid chunks with IDs and content to add to Chroma.")
             return

        existing_items = db.get(include=[]) # Fetch only IDs
        existing_ids = set(existing_items["ids"])
        st.sidebar.write(f"Number of existing documents in DB: {len(existing_ids)}")

        new_chunks = [
            chunk for chunk in valid_chunks if chunk.metadata["id"] not in existing_ids
        ]

        if new_chunks:
            st.sidebar.write(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            # Add documents in batches
            batch_size = 32 # Adjust batch size as needed
            added_count = 0
            with st.spinner(f"Adding {len(new_chunks)} chunks to database..."):
                 for i in range(0, len(new_chunks), batch_size):
                     batch = new_chunks[i : i + batch_size]
                     batch_ids = new_chunk_ids[i : i + batch_size]
                     try:
                         db.add_documents(documents=batch, ids=batch_ids)
                         added_count += len(batch)
                     except Exception as batch_error:
                         st.sidebar.error(f"Error adding batch {i // batch_size + 1}: {batch_error}")
                         # Optionally continue to next batch or stop
            st.sidebar.success(f"✅ Successfully added {added_count} new chunks.")
            # Optional: Persist DB changes explicitly if needed, though Chroma usually handles this
            # db.persist()
        else:
            st.sidebar.write("✅ No new documents to add.")

    except Exception as e:
         st.sidebar.error(f"❌ Failed to initialize or add to ChromaDB: {e}")
         import traceback
         st.sidebar.text(traceback.format_exc())


##############################
# RAG Function
##############################
@st.cache_resource # Cache the model and tokenizer
def load_gemma_model():
    # No need to check session_state here due to @st.cache_resource
    load_dotenv()
    MODEL_LLM = os.getenv("MODEL_LLM", "google/gemma-3-4b-it")
    MODEL_DIR = os.getenv("MODEL_DIR", "./model")

    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        st.sidebar.info(f"Downloading model {MODEL_LLM} to {MODEL_DIR}...")
        try:
            with st.spinner("Downloading LLM (this may take a while)..."):
                 snapshot_download(
                    repo_id=MODEL_LLM,
                    local_dir=MODEL_DIR,
                    local_dir_use_symlinks=False, # Avoid symlink issues
                    resume_download=True,
                    #ignore_patterns=["*.safetensors.index.json", "*.gguf"], # Optional: Ignore specific large files if not needed
                )
            st.sidebar.success(f"Model downloaded successfully")
        except Exception as e:
            st.sidebar.error(f"Error downloading model: {e}")
            st.error(f"Failed to download the AI model. Please check the console/logs. Error: {e}")
            return None, None # Return None if download fails

    try:
        with st.spinner("🤖 Loading AI model..."):
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.sidebar.write(f"Using device: {device}")

            model = Gemma3ForCausalLM.from_pretrained(
                 MODEL_DIR, # Load from local directory
                 torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32, # Use bfloat16 only on CUDA
                 device_map="auto" # Let transformers handle device placement
                 # low_cpu_mem_usage=True # May help on systems with limited RAM
            )

            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR) # Load from local dir
        st.sidebar.success("AI Model loaded.")
        return model, tokenizer
    except Exception as e:
        st.sidebar.error(f"Error loading model from {MODEL_DIR}: {e}")
        st.error(f"Failed to load the AI model. Please check the console/logs. Error: {e}")
        import traceback
        st.sidebar.text(traceback.format_exc())
        return None, None # Return None if loading fails


def generate_text(model, tokenizer, prompt, max_new_tokens=8192):
    """
    Generate text using Gemma-3 model and tokenizer
    """
    if model is None or tokenizer is None:
        return "Error: Model or tokenizer not loaded."

    try:
        # Format for Gemma-3 chat format
        if hasattr(tokenizer, "apply_chat_template"):
            # Structure for Gemma-3 chat: list of turns, each turn is a dict with role/content
            # The content itself is a list of parts (here just one text part)
            messages = [
                 {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            # Important: Do NOT add generation prompt for Gemma-3's own template
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False, # Gemma-3's template usually includes indicators
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            # Add the generation prompt token(s) MANUALLY if needed after templating
            # Check tokenizer docs for specific start-of-generation tokens if required.
            # For many instruction-tuned models, the template handles this.

            input_len = inputs["input_ids"].shape[-1]

            # Generation parameters
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id # Important for generation
            )

            # Decode only the newly generated tokens
            response = tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )
        else:
             # Fallback (less ideal for chat models)
             st.warning("Tokenizer does not have apply_chat_template. Using basic encoding.")
             input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
             input_len = input_ids.shape[-1]
             outputs = model.generate(
                 input_ids,
                 max_new_tokens=max_new_tokens,
                 do_sample=True,
                 temperature=0.7,
                 top_k=50,
                 top_p=0.95,
                 pad_token_id=tokenizer.eos_token_id
             )
             response = tokenizer.decode(
                 outputs[0][input_len:], skip_special_tokens=True
             )

        return response.strip() # Clean up whitespace

    except Exception as e:
        st.error(f"Error generating text: {e}")
        import traceback
        st.text(traceback.format_exc())
        return f"Error generating response: {str(e)}"

##############################
# STREAMLIT LAYOUT
##############################
def main():
    ensure_directories_exist()
    st.title("Chat")
    st.sidebar.title("Your Sources")

    # Choose mode
    mode = st.sidebar.radio(
        "⚙️ Select Mode",
        ["📚 RAG Mode (with Documents)", "✨ Pure Gemma-3"],
        index=0,
        help="RAG mode uses document context, Pure mode answers generally",
    )

    # Initialize session_state if not already done
    if "language" not in st.session_state:
        st.session_state.language = "English"
    if "messages" not in st.session_state:
         st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Tôi có thể giúp gì bạn hôm nay?"}]
    if "initialized" not in st.session_state:
         st.session_state.initialized = True # Mark as initialized
    if "prev_mode" not in st.session_state:
        st.session_state.prev_mode = mode


    # Mode Change Welcome Message Logic
    if mode != st.session_state.prev_mode:
        mode_welcome = {
            "📚 RAG Mode (with Documents)": "Chế độ RAG đã được kích hoạt. Hãy hỏi về tài liệu đã tải lên.",
            "✨ Pure Gemma-3": "Chế độ trả lời tổng quát đã được kích hoạt. Hãy hỏi bất cứ điều gì!",
        }
        # Reset chat history with new welcome message
        st.session_state.messages = [{"role": "assistant", "content": mode_welcome[mode]}]
        st.session_state.prev_mode = mode
        # Force rerun to display the new message immediately
        st.rerun()


    # --- Sidebar Controls ---
    if st.sidebar.button("Clear Database", key="clear_db"):
        clear_database()

    # Language selection and file upload logic
    prev_language = st.session_state.language
    st.session_state.language = st.sidebar.radio(
        "Select Document Language",
        ("English", "Vietnamese"),
        index=0 if st.session_state.language == "English" else 1,
        key="lang_select"
    )

    # Use a consistent key for the uploader, reset if language changes
    uploader_key = f"file_uploader_{st.session_state.language}"
    if prev_language != st.session_state.language:
        # Clear file uploader state for the NEW language if switching
        if uploader_key in st.session_state:
            # st.session_state[uploader_key] = [] # Reset to empty list
            del st.session_state[uploader_key] # Reset to empty list

    uploaded_files = st.sidebar.file_uploader(
        "Add source PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=uploader_key, # Use the language-specific key
    )

    if uploaded_files:
        target_dir = DATA_PATH_EN if st.session_state.language == "English" else DATA_PATH_VI
        files_saved = 0
        for uploaded_file in uploaded_files:
            file_path = os.path.join(target_dir, uploaded_file.name)
            file_exists = os.path.exists(file_path)
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if file_exists:
                    st.sidebar.info(f"Overwrote existing file: {uploaded_file.name}")
                else:
                    st.sidebar.success(f"Saved {uploaded_file.name} to {target_dir}")
                files_saved += 1
            except Exception as e:
                st.sidebar.error(f"Failed to save {uploaded_file.name}: {e}")
        # Clear the uploader state after processing IF files were saved
        # This prevents reprocessing the same files on rerun unless new ones are added
        if files_saved > 0:
             # st.session_state[uploader_key] = []
             del st.session_state[uploader_key] # Reset to empty list


    if st.sidebar.button("Process Documents & Build DB", key="process_docs"):
        with st.spinner("Processing documents and updating database..."):
            try:
                docs = load_all_documents()
                if docs:
                    chunks = chunk_documents(docs)
                    if chunks:
                         add_to_chroma(chunks)
                         st.sidebar.success("✅ Database processing complete.")
                    else:
                         st.sidebar.warning("⚠️ No chunks created from documents.")
                else:
                    st.sidebar.error("No valid documents found to process.")
            except Exception as e:
                st.sidebar.error(f"❌ Error during database processing: {e}")
                import traceback
                st.sidebar.text(traceback.format_exc())

    # =================== Chat UI =======================
    st.divider()
    st.subheader("💬 Chat with Gemma-3")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input - This triggers a rerun when input is submitted
    if prompt := st.chat_input("Ask anything...."):
        # Add user message to history immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message immediately (Streamlit handles the rerun for this)
        with st.chat_message("user"):
            st.markdown(prompt)

        # *** START: Processing Block - ONLY runs when prompt is NEW ***
        start_time = time.time()
        response_content = "..." # Placeholder

        # Display thinking indicator
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Thinking...")

            try:
                # Load model (cached) - Ensure it's loaded successfully
                model, tokenizer = load_gemma_model()
                if model is None or tokenizer is None:
                    st.error("Model not loaded. Cannot generate response.")
                    st.stop() # Stop execution if model isn't ready

                # --- RAG Mode Logic ---
                if mode == "📚 RAG Mode (with Documents)":
                    # Check DB existence
                    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
                         response_content = "⚠️ Database is empty or not found. Please upload and process documents first."
                    else:
                        embedding_func = get_embedding()
                        if embedding_func is None:
                             response_content = "❌ Error: Embedding function not available."
                        else:
                            try:
                                 db = Chroma(
                                     collection_name="RAGDB",
                                     persist_directory=CHROMA_PATH,
                                     embedding_function=embedding_func
                                 )
                                 # Perform similarity search
                                 results = db.similarity_search_with_score(prompt, k=3)

                                 # Check if results were found and filter out None content
                                 valid_results = [(doc, score) for doc, score in results if doc.page_content is not None]

                                 if valid_results:
                                     # Build context safely
                                     context = "\n\n---\n\n".join([doc.page_content for doc, _score in valid_results])

                                     # Limit context length (optional, consider model's limit)
                                     # if len(context) > 10000:
                                     #     context = context[:10000] + "..."

                                     # Create the RAG prompt
                                     rag_prompt = f"""Dựa vào ngữ cảnh sau đây, hãy trả lời câu hỏi một cách chi tiết và rõ ràng bằng tiếng Việt:
                                        Ngữ cảnh:
                                        {context}

                                    Câu hỏi: {prompt}

                                    Hãy cung cấp một câu trả lời chi tiết, giải thích rõ ràng các khái niệm và ý tưởng liên quan.
                                    Sử dụng các đoạn văn để trình bày thông tin một cách có cấu trúc.
                                    Đảm bảo rằng câu trả lời của bạn chỉ dựa trên thông tin từ ngữ cảnh được cung cấp."""

                                     # Generate response using the RAG prompt
                                     response_content = generate_text(model, tokenizer, rag_prompt)

                                     # Optionally add sources (ensure metadata['id'] exists)
                                     sources = [doc.metadata.get("id", "Unknown Source") for doc, _score in valid_results]
                                     if sources:
                                         response_content += f"\n\n📚 Nguồn tham khảo:\n" + "\n".join(f"- `{s}`" for s in set(sources))

                                 else:
                                     # No relevant documents found
                                     response_content = "Rất tiếc, tôi không tìm thấy thông tin liên quan trong tài liệu của bạn. Bạn có muốn tôi thử trả lời dựa trên kiến thức chung không?"
                                     # Optionally: Fallback to pure generation here if desired
                                     # pure_prompt = f"..."
                                     # response_content = generate_text(model, tokenizer, pure_prompt)

                            except Exception as db_error:
                                 response_content = f"❌ Lỗi khi truy vấn cơ sở dữ liệu: {db_error}"
                                 st.error(f"Database query error: {db_error}")
                                 import traceback
                                 st.text(traceback.format_exc())

                # --- Pure Gemma-3 Mode Logic ---
                else:
                     pure_prompt = f"""Bạn là một trợ lý ảo rất giỏi.
                    Hãy cung cấp một câu trả lời chi tiết, giải thích rõ ràng các khái niệm và ý tưởng liên quan.
                    Sử dụng các đoạn văn để trình bày thông tin một cách có cấu trúc.
                    Câu hỏi: {prompt}
                    Trả lời:"""
                     response_content = generate_text(model, tokenizer, pure_prompt)

            except Exception as e:
                 response_content = f"❌ Đã xảy ra lỗi trong quá trình xử lý: {str(e)}"
                 st.error(f"Processing error: {e}")
                 import traceback
                 st.text(traceback.format_exc()) # Log full traceback

            # Calculate execution time
            end_time = time.time()
            execution_time = end_time - start_time

            # Add execution time to the response
            response_content += f"\n\n⏱️ *Thời gian xử lý: {execution_time:.2f} giây*"

            # Update the placeholder with the final response
            message_placeholder.markdown(response_content)

            # Add final assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": response_content})

        # *** END: Processing Block ***


if __name__ == "__main__":
    # Load environment variables (optional, if needed globally)
    # load_dotenv()
    main()
