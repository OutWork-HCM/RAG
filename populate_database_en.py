from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from utils import test_chunks, inspect_chunk_structure, determine_optimal_configuration
import os
import argparse
import shutil
from get_embedding import get_embedding
from tqdm import tqdm


CHROMA_PATH = "./myDB"  # this folder is used to store ChromaDB files
DATA_PATH = "./data/en"  # this folder is used to store all PDFs


def load_document():
    """
    Load all PDFs from the data directory
    """
    documents = []
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise ValueError(f"No PDF files found in {DATA_PATH}")

    print(f"Found {len(pdf_files)} PDF files in {DATA_PATH}")

    for pdf_file in tqdm(pdf_files, desc="Loading PDF files"):
        file_path = os.path.join(DATA_PATH, pdf_file)
        try:
            document_loader = PyMuPDFLoader(file_path=file_path)
            doc = document_loader.load()
            documents.extend(doc)
            print(f"✅ Loaded {pdf_file}: {len(doc)} pages")
        except Exception as e:
            print(f"❌ Error loading {pdf_file}: {str(e)}")

    print(f"Total pages load: {len(documents)}")
    return documents


def split_document(documents: list[Document]):
    """
    Split documents into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False
    )
    print(f"Splitting {len(documents)} pages into chunks")
    chunks = text_splitter.split_documents(documents)
    print(f"Create {len(chunks)} chunks")
    return chunks


def calculate_chunk_ids(chunks):
    """
    Create unique IDs for each chunk based on source, page, and chunk index

    This is create IDs like: "data/file.pfd:6:2"
    Page Source: Page Number: Chunk Index
    """
    page_chunk_counts = {}
    for chunk in tqdm(chunks, desc="Processing chunks"):
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        # Init counter if this is new page
        if current_page_id not in page_chunk_counts:
            page_chunk_counts[current_page_id] = 0
        else:
            page_chunk_counts[current_page_id] += 1

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{page_chunk_counts[current_page_id]}"

        # Add it to the page meta-data
        chunk.metadata["id"] = chunk_id
    return chunks


def clear_database():
    """
    Clear teh existing Chroma database
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Deleted existing database at {CHROMA_PATH}")


def add_to_chroma(chunks: list[Document]):
    """
    Add chunks to Chromadb, avoiding duplicates
    """
    # Load the exist database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding(),
    )

    # Calculate Page IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents taht don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")

        # Process in batches to show progress
        batch_size = 16
        for i in tqdm(range(0, len(new_chunks), batch_size), desc="Adding to database"):
            batch_chunks = new_chunks[i : i + batch_size]
            batch_ids = [chunk.metadata["id"] for chunk in batch_chunks]
            db.add_documents(batch_chunks, ids=batch_ids)
    else:
        print("✅ No new documents to add")


def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update the data store)
    try:
        document = load_document()
        chunks = split_document(document)
        add_to_chroma(chunks)
        print("✅ Database population completed successfully")
    except Exception as e:
        print(f"❌ Error during database population: {str(e)}")
        raise

    # PERF:debugpoint
    # In một số chunk đầu tiên để kiểm tra
    # test_chunks(chunks, num_samples=3, full_chunk_index=5, verbose=True)
    # Kiểm tra câu trúc chunks
    # inspect_chunk_structure(chunks)


if __name__ == "__main__":
    main()
