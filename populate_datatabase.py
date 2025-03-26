# Library
from get_embedding import get_embedding
import os
import argparse
import shutil
import PyPDF2
from tqdm import tqdm
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from langchain.schema.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Directory
CHROMA_PATH = "./myDB"  # Folder for ChromaDB
DATA_PATH_VI = "./data/vi"  # Vietnamese PDFs
DATA_PATH_EN = "./data/en"  # English PDFs


def ensure_directories_exist():
    """
    Ensure required directories exist. If not, create them.
    """
    for path in [CHROMA_PATH, DATA_PATH_EN, DATA_PATH_VI]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"📁 Created directory: {path}")
        else:
            print(f"✅ Directory exists: {path}")


##############################################################
# For Vietnamese PDFs: using marker-pdf and PyPDF2
##############################################################
def get_pdf_metadata(file_path: str) -> dict:
    """
    Get metadata of PDF by PyPDF2
    Return empty dictionary if have any error.
    """
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
        print(f"Error extracting metadata from {file_path}: {e}")
    return meta_dict


def split_into_pages(text: str, total_pages: int) -> list:
    """
    Split the document content into pages based on the desired number of pages.
    When a fixed split position (by length) does not end with a period (.) or ! or ?,
    the function will look for the next period to break the page.
    If no period is found in the remaining content, the fixed split position will be used.
    """
    if total_pages <= 0:
        return [text]
    text_length = len(text)
    approx_page_length = text_length // total_pages
    pages = []

    start = 0
    for i in range(total_pages):
        # If the last page, take all context
        if i == total_pages - 1:
            pages.append(text[start:])
            break

        # Indentify location of break point
        end = start + approx_page_length
        if end >= text_length:
            pages.append(text[start:])
            break

        # If there is no period (.) at the cut position, search for the next period.
        while end < text_length and text[end] not in ".!?":
            end += 1

        # Found the period
        if end < text_length:
            end += 1

        pages.append(text[start:end])
        start = end
    return pages


def load_vi_documents() -> list[Document]:
    """
    Load all PDF files from DATA_PATH_VI.
    - Get the original metadata of the file using PyPDF2.
    - Use marker-pdf to extract the content of the PDF file (with results in markdown format and metadata).
    - Split the text into pages based on the metadata.
    - Assign metadata (including file name, page number, …) to each page and create a Document object.
    """
    # marker-pdf option
    # To see all available options, do marker_single --help
    config = {
        "output_format": "markdown",  # [markdown|json|html]
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
        raise ValueError(f"No Vietnamese PDF files found in {DATA_PATH_VI}")

    print(f"Found {len(pdf_files)} Vietnamese PDF files in {DATA_PATH_VI}")

    for pdf_file in tqdm(pdf_files, desc="Loading Vietnamese PDF files"):
        file_path = os.path.join(DATA_PATH_VI, pdf_file)
        try:
            pdf_meta = get_pdf_metadata(file_path)
            total_pages = pdf_meta.get("total_pages", 1)

            # Use markder-pdf to extract context of PDF file
            rendered = converter(file_path)
            extracted = text_from_rendered(rendered)
            text = extracted[0]  # Markdown text
            page_stats = rendered.metadata.get("page_stats", [])
            if page_stats:
                total_pages = max(total_pages, len(page_stats))
            print(f"✅ Loaded {pdf_file}: {total_pages} pages")

            # Split paragraph into pages
            page_texts = split_into_pages(text, total_pages)

            for i, page_text in enumerate(page_texts):
                page_meta = pdf_meta.copy()
                page_meta["page"] = i
                doc = Document(page_content=page_text, metadata=page_meta)
                documents.append(doc)
        except Exception as e:
            print(f"❌ Error loading {pdf_file}: {str(e)}")
    print(f"Total Vietnamese pages load: {len(documents)}")
    return documents


##############################################################
# For English PDFs: using PyMuPDF
##############################################################
def load_en_documents():
    """
    Load all English PDFs from the DATA_PATH_EN directory by PyMuPDF
    """
    documents = []
    pdf_files = [f for f in os.listdir(DATA_PATH_EN) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise ValueError(f"No English PDF files found in {DATA_PATH_EN}")

    print(f"Found {len(pdf_files)} English PDF files in {DATA_PATH_EN}")

    for pdf_file in tqdm(pdf_files, desc="Loading English PDF files"):
        file_path = os.path.join(DATA_PATH_EN, pdf_file)
        try:
            document_loader = PyMuPDFLoader(file_path=file_path)
            doc = document_loader.load()
            documents.extend(doc)
            print(f"✅ Loaded {pdf_file}: {len(doc)} pages")
        except Exception as e:
            print(f"❌ Error loading {pdf_file}: {str(e)}")

    print(f"Total English pages load: {len(documents)}")
    return documents


##############################################################
# Merge all documents
##############################################################
def load_all_documents() -> list[Document]:
    """
    Load all documents (English and Vietnamese) and merge together.
    """
    docs_en = load_en_documents()
    docs_vi = load_vi_documents()
    if not docs_en and not docs_vi:
        raise ValueError(
            "No PDF files found in both English and Vietnamese directories."
        )
    elif not docs_en:
        raise ValueError(f"No English PDF files found in {DATA_PATH_EN}")
    elif not docs_vi:
        raise ValueError(f"No Vietnamese PDF files found in {DATA_PATH_VI}")

    return docs_en + docs_vi


##############################################################
# Chunking and ID assignment
##############################################################
def chunk_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False
    )
    print(f"Splitting {len(documents)} pages into chunks")
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """
    Assign a unique ID to each chunk in the format: "source:page:chunk_index"
    """
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


##############################################################
# python polulate_database.py --reset
##############################################################
def clear_database():
    """
    Delete current ChromaDB.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Deleted existing database at {CHROMA_PATH}")


##############################################################
# Add embedding datas into ChromaDB
##############################################################
def add_to_chroma(chunks: list[Document]):
    """
    Add chunks to ChromaDB, ignoring duplicate Documents.
    """
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding(),
    )
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [
        chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids
    ]
    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        batch_size = 16
        for i in tqdm(range(0, len(new_chunks), batch_size), desc="Adding to database"):
            batch_chunks = new_chunks[i : i + batch_size]
            batch_ids = [chunk.metadata["id"] for chunk in batch_chunks]
            db.add_documents(batch_chunks, ids=batch_ids)
    else:
        print("✅ No new documents to add")


##############################################################
# main function
##############################################################
def main():
    # Make sure directories exists
    ensure_directories_exist()

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    try:
        docs = load_all_documents()
        chunks = chunk_documents(docs)
        add_to_chroma(chunks)
        print("✅ Database population completed successfully")
    except Exception as e:
        print(f"❌ Error during database population: {str(e)}")
        raise


if __name__ == "__main__":
    main()
