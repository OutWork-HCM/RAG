import streamlit as st
import os
import shutil
from get_embedding import get_embedding
import PyPDF2
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

# FIXME: Error: 'NoneType' object has no attribute 'replace' in "RAG Mode"

# --- Thư mục và hàm xử lý của bạn ---
CHROMA_PATH = "./myDB"  # Folder for ChromaDB
DATA_PATH_EN = "./data/en"  # English PDFs
DATA_PATH_VI = "./data/vi"  # Vietnamese PDFs


def ensure_directories_exist():
    for path in [CHROMA_PATH, DATA_PATH_EN, DATA_PATH_VI]:
        if not os.path.exists(path):
            os.makedirs(path)
            # st.sidebar.write(f"📁 Created directory: {path}")
        # else:
        #     st.sidebar.write(f"✅ Directory exists: {path}")


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        st.sidebar.success(f"Deleted existing database at {CHROMA_PATH}")


def get_pdf_metadata(file_path: str) -> dict:
    meta_dict = {}
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            meta = reader.metadata
            total_pages = len(reader.pages)
            meta_dict = {
                "producer": meta.get("/Producer", ""),  # pyright: ignore
                "creator": meta.get("/Creator", ""),  # pyright: ignore
                "creationdate": meta.get("/CreationDate", ""),  # pyright: ignore
                "source": file_path,
                "file_name": os.path.basename(file_path),
                "total_pages": total_pages,
                "format": meta.get("/PDFFormat", "PDF 1.x"),  # pyright: ignore
                "title": meta.get("/Title", ""),  # pyright: ignore
                "author": meta.get("/Author", ""),  # pyright: ignore
                "subject": meta.get("/Subject", ""),  # pyright: ignore
                "keywords": meta.get("/Keywords", ""),  # pyright: ignore
                "moddate": meta.get("/ModDate", ""),  # pyright: ignore
            }
    except Exception as e:
        st.sidebar.error(f"Error extracting metadata from {file_path}: {e}")
    return meta_dict


def split_into_pages(text: str, total_pages: int) -> list:
    if total_pages <= 0:
        return [text]
    text_length = len(text)
    approx_page_length = text_length // total_pages
    pages = []
    start = 0
    for i in range(total_pages):
        if i == total_pages - 1:
            pages.append(text[start:])
            break
        end = start + approx_page_length
        if end >= text_length:
            pages.append(text[start:])
            break
        while end < text_length and text[end] not in ".!?":
            end += 1
        if end < text_length:
            end += 1
        pages.append(text[start:end])
        start = end
    return pages


def load_vi_documents() -> list[Document]:
    config = {
        "output_format": "markdown",
        "force_ocr": True,
        "languages": "vi,en",
    }
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
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
            total_pages = pdf_meta.get("total_pages", 1)
            rendered = converter(file_path)
            extracted = text_from_rendered(rendered)
            text = extracted[0]
            page_stats = rendered.metadata.get("page_stats", [])
            if page_stats:
                total_pages = max(total_pages, len(page_stats))
            st.sidebar.write(f"✅ Loaded {pdf_file}: {total_pages} pages")
            page_texts = split_into_pages(text, total_pages)
            for i, page_text in enumerate(page_texts):
                page_meta = pdf_meta.copy()
                page_meta["page"] = i
                doc = Document(page_content=page_text, metadata=page_meta)
                documents.append(doc)
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {pdf_file}: {str(e)}")
    st.sidebar.write(f"Total Vietnamese pages load: {len(documents)}")
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
            docs = loader.load()
            documents.extend(docs)
            st.sidebar.write(f"✅ Loaded {pdf_file}: {len(docs)} pages")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {pdf_file}: {str(e)}")
    st.sidebar.write(f"Total English pages load: {len(documents)}")
    return documents


def load_all_documents() -> list[Document]:
    docs_en = load_en_documents()
    docs_vi = load_vi_documents()
    # Kiểm tra tình trạng rỗng
    if not docs_en and not docs_vi:
        st.sidebar.error(
            "No PDF files found in both English and Vietnamese directories."
        )
        return []
    elif not docs_en:
        st.sidebar.error(f"No English PDF files found in {DATA_PATH_EN}")
    elif not docs_vi:
        st.sidebar.error(f"No Vietnamese PDF files found in {DATA_PATH_VI}")
    return docs_en + docs_vi


def chunk_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False
    )
    st.sidebar.write(f"Splitting {len(documents)} pages into chunks")
    chunks = text_splitter.split_documents(documents)
    st.sidebar.write(f"Created {len(chunks)} chunks")
    return chunks


def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    page_chunk_counts = {}
    for chunk in tqdm(chunks, desc="Processing chunks"):
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        page_chunk_counts[current_page_id] = page_chunk_counts.get(current_page_id, 0)
        chunk_id = f"{current_page_id}:{page_chunk_counts[current_page_id]}"
        page_chunk_counts[current_page_id] += 1
        chunk.metadata["id"] = chunk_id
    return chunks


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding(),
    )
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    st.sidebar.write(f"Number of existing documents in DB: {len(existing_ids)}")
    new_chunks = [
        chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids
    ]
    if new_chunks:
        st.sidebar.write(f"👉 Adding new documents: {len(new_chunks)}")
        batch_size = 16
        for i in range(0, len(new_chunks), batch_size):
            batch_chunks = new_chunks[i : i + batch_size]
            batch_ids = [chunk.metadata["id"] for chunk in batch_chunks]
            db.add_documents(batch_chunks, ids=batch_ids)
    else:
        st.sidebar.write("✅ No new documents to add")


##############################
# RAG Function
##############################
def load_gemma_model():
    # Check model is loaded?
    if "gemma_model" not in st.session_state:
        load_dotenv()
        MODEL_LLM = "google/gemma-3-4b-it"  # Read this page how to use https://huggingface.co/blog/gemma3
        MODEL_DIR = os.getenv("MODEL_DIR", "./model")

        # Create folder in not exists
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Download model if not exists
        if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
            print(f"Downloading model to {MODEL_DIR}...")
            try:
                snapshot_download(
                    repo_id=MODEL_LLM,
                    local_dir=MODEL_DIR,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                print(f"Model downloaded successfully")
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise

        # Load model with session_state
        with st.spinner("🤖 Đang khởi động AI..."):
            st.session_state.gemma_model = Gemma3ForCausalLM.from_pretrained(
                MODEL_LLM, torch_dtype=torch.bfloat16, device_map="auto"
            )
            st.session_state.gemma_tokenizer = AutoTokenizer.from_pretrained(MODEL_LLM)
    return st.session_state.gemma_model, st.session_state.gemma_tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=512):
    """
    Generate text using Gemma-3 model and tokenizer
    """
    try:
        # Format for Gemma-3 chat format
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            # Generate output
            input_len = inputs["input_ids"].shape[-1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )

            # Decode and return only the new tokens
            response = tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )
        else:
            # Fallback to standard generation if chat template is not available
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                model.device
            )
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
            response = tokenizer.decode(
                outputs[0][input_ids.shape[1] :], skip_special_tokens=True
            )

        return response
    except Exception as e:
        print(f"Error generating text: {e}")
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

    # Khởi tạo session_state cho ngôn ngữ
    if "language" not in st.session_state:
        st.session_state.language = "English"

    # Nút Clear Database
    if st.sidebar.button("Clear Database"):
        clear_database()  # Gọi hàm xóa DB của bạn
        st.sidebar.success("Database cleared.")

    # Radio chọn ngôn ngữ - CẬP NHẬT TRỰC TIẾP SESSION STATE
    prev_language = st.session_state.language
    st.session_state.language = st.sidebar.radio(
        "Select Document Language",
        ("English", "Vietnamese"),
        index=0 if st.session_state.language == "English" else 1,
    )

    # Tạo key cho file uploaded dựa trên ngôn ngữ
    uploaded_key = f"file_uploader_{st.session_state.language}"

    # Reset uploaded files khi ngon ngu thay doi
    if prev_language != st.session_state.language:
        if uploaded_key in st.session_state:
            del st.session_state[uploaded_key]

    # Widget upload file - KHÔNG DÙNG KEY TRÙNG VỚI SESSION STATE
    uploaded_files = st.sidebar.file_uploader(
        "Add source",
        type=["pdf"],
        accept_multiple_files=True,
        key=uploaded_key,
    )

    # Xử lý khi có file upload
    if uploaded_files:
        target_dir = (
            DATA_PATH_EN if st.session_state.language == "English" else DATA_PATH_VI
        )

        # Save file to folder
        for uploaded_file in uploaded_files:
            file_path = os.path.join(target_dir, uploaded_file.name)

            # Kiem tra su trung lap
            if os.path.exists(file_path):
                st.sidebar.warning(
                    f"File {uploaded_file.name} already exists, overwriting..."
                )

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"Saved {uploaded_file.name} to {target_dir}")

    # Nút xử lý Documents (embedding,...)
    if st.sidebar.button("Process Documents"):
        try:
            docs = load_all_documents()
            os.makedirs(CHROMA_PATH, exist_ok=True)
            if docs:
                chunks = chunk_documents(docs)
                add_to_chroma(chunks)
                st.sidebar.success("✅ Database population completed successfully")
            else:
                st.sidebar.error("No documents to process.")
        except Exception as e:
            st.sidebar.error(f"❌ Error during database population: {e}")

    # =================== RAG UI =======================
    st.divider()
    st.subheader("💬 Chat with Gemma-3")

    # Kiểm tra lần đầu load app
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        welcome_msg = "Chào bạn! Tôi có thể giúp gì bạn hôm nay?"
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

    # Kiểm tra thay đổi chế độ
    if "prev_mode" not in st.session_state:
        st.session_state.prev_mode = mode

    if mode != st.session_state.prev_mode:
        # Thêm lời chào khi chuyển chế độ
        mode_welcome = {
            "📚 RAG Mode (with Documents)": "Chào bạn! Tôi đã sẵn sàng phân tích tài liệu của bạn. Hãy hỏi tôi bất cứ điều gì về các file PDF bạn đã upload!",
            "✨ Pure Gemma-3": "Xin chào! Tôi có thể trả lời mọi câu hỏi tổng quát. Hãy hỏi tôi bất cứ điều gì!",
        }
        st.session_state.messages = [
            {"role": "assistant", "content": mode_welcome[mode]}
        ]
        st.session_state.prev_mode = mode

    # Instance session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Question
    if prompt := st.chat_input("Ask anything...."):
        # Add question into history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Record timing
    start_time = time.time()

    # Process
    with st.spinner("Processing......."):
        try:
            # Load model and database
            model, tokenizer = load_gemma_model()

            # Work with RAG
            if mode == "📚 RAG Mode (with Documents)":
                # Check DB exist?
                if not os.path.exists(CHROMA_PATH):
                    st.error("Database not found! Please process documents first.")
                    st.stop()
                # RAG Process
                db = Chroma(
                    persist_directory=CHROMA_PATH, embedding_function=get_embedding()
                )
                # search the DB - Retrieval
                results = db.similarity_search_with_score(
                    prompt, k=3  # pyright: ignore
                )

                context = "\n\n---\n\n".join(
                    [doc.page_content for doc, _score in results]
                )

                if len(context) > 10000:
                    context = context[:10000] + "..."

                # Collect sources
                # sources = [doc.metadata.get("id", None) for doc, _score in results]

                # Create prompt
                # rag_prompt = f"""Answer based ONLY on this context:
                # {context}
                #     
                # Question: {prompt}
                # Answer in Vietnamese clearly and concisely:"""
                rag_prompt = f"""Dựa vào ngữ cảnh sau đây, hãy trả lời câu hỏi một cách chi tiết và rõ ràng bằng tiếng Việt: 
                Ngữ cảnh:
                {context}

                Câu hỏi: {prompt}

                Hãy cung cấp một câu trả lời chi tiết, giải thích rõ ràng các khái niệm và ý tưởng liên quan. 
                Sử dụng các đoạn văn để trình bày thông tin một cách có cấu trúc. 
                Đảm bảo rằng câu trả lời của bạn chỉ dựa trên thông tin từ ngữ cảnh được cung cấp."""
                # Answer
                response = generate_text(model, tokenizer, rag_prompt)
                # add source to referencing
                # response += f"\n\n📚 Sources:\n" + "\n".join(set(sources))
            else:  # Pure Gemma-3 mode
                # pure_prompt = f"""You are a helpful AI assistant. 
                #     Answer the following question clearly and concisely in Vietnamese:
                #     
                #     Question: {prompt}
                #     Answer:"""
                pure_prompt = f"""Bạn là một trợ lý ảo rất giỏi.
                Hãy cung cấp một câu trả lời chi tiết, giải thích rõ ràng các khái niệm và ý tưởng liên quan. 
                Sử dụng các đoạn văn để trình bày thông tin một cách có cấu trúc.
                Question: {prompt}
                Answer:"""
                response = generate_text(model, tokenizer, pure_prompt)
        except Exception as e:
            response = f"Error: {str(e)}"

    # Record timing
    end_time = time.time()
    execution_time = end_time - start_time
    # Hiển thị câu trả lời
    with st.chat_message("assistant"):
        st.markdown(response)
        st.markdown(f"**Thời gian xử lý: {execution_time:.2f} giây**")
    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
