import os
import fitz  # pymupdf is better than pypdf, requires 'conda install conda-forge::pymupdf'
from tqdm.auto import (
    tqdm,
)  # progress bar display, requires 'conda install conda-forge::tqdm'
from spacy.lang.en import English  # see https://spacy.io/usage
import chromadb  # requires 'conda install conda-forge::chromadb'
from sentence_transformers import SentenceTransformer
import torch
import re
import random
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# PDF list
pdf_files = ["./PDFs/VerilogHDL_AGuide2DigitalDesignAndSynthesis.pdf"]

# Check if the directory ./myDB exist or not
if not os.path.exists("./myDB"):
    os.makedirs("./myDB")


# Clean text by removing newlines and extra whitespace
def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()


# Open and read PDF files, return lines/pages
def open_and_read_pdf(pdf_file: str) -> list[dict]:
    """
    Open PDF file, read its content page by page, and collects statistics

    Parameters:
        pdf_file: (str): the file path to pdf file to be opened and read.

    Return:
        list[dict]: a listof dictionaries, contain:
            + page number
            + extracted text for each page
    """
    doc = fitz.open(pdf_file)  # Open pdf file
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # pyright: ignore
        text = page.get_text()  # read text in a page
        text = text_formatter(text)  # format
        pages_and_texts.append({"page_number": page_number + 1, "text": text})
    return pages_and_texts


# Split text into sentences using spaCy
def split_into_sentences(text: str, nlp: English) -> list[str]:
    doc = nlp(text)  # Process text by spaCy
    return [
        str(sent) for sent in doc.sents
    ]  # return list of sentences in string format


# PERF: debug point
# text = """
# As you might've guessed, this process is referred to as chunking.
# Why do we do this?
#    1 - Easier to manage similar sized chunks of text.
#    2 - Don't overload the embedding models capacity for tokens (e.g. if an embedding model has a capacity of 384 tokens, there could be information loss if you try to embed a sequence of 400+ tokens).
#    3 - Our LLM context window (the amount of tokens an LLM can take in) may be limited and requires compute power so we want to make sure we're using it as well as possible.
# Something to note is that there are many different ways emerging for creating chunks of information/text.
# For now, we're going to keep it simple and break our pages of sentences into groups of 10 (this number is arbitrary and can be changed, I just picked it because it seemed to line up well with our embedding model capacity of 384).
# On average each of our pages has 10 sentences, and an average total of 287 tokens per page. So our groups of 10 sentences will also be ~287 tokens long.
# This gives us plenty of room for the text to embedded by our all-mpnet-base-v2 model (it has a capacity of 384 tokens).
# """
# nlp = English()
# nlp.add_pipe("sentencizer")
# sentences = split_into_sentences(text=text, nlp=nlp)
# for sentence in sentences:
#     sentence = sentence.strip()
#     print(sentence)


# chunking sentences together
def chunk_sentences(sentences: list[str], chunk_size: int) -> list[list[str]]:
    return [sentences[i : i + chunk_size] for i in range(0, len(sentences), chunk_size)]


# progress a pdf file and return a chunk of text
def process_pdf(pdf_path: str, nlp: English, chunk_size: int) -> list[dict]:
    """
    Processes a PDF file by reading its text, splitting it into sentences, and grouping them into chunks.


    Parameters:
        nlp (spacy.lang.en.English): A spaCy English language model used to split the text into sentences.
        chunk_size (int): The number of sentences per chunk.

    Returns:
        list[dict]: A list of dictionaries, each representing a chunk of text with the following keys:
            - filename (str): The name of the PDF file.
            - page_number (int): The page number from which the chunk comes.
            - sentence_chunk (str): The text of the chunk, with formatting issues corrected.
            - chunk_char_count (int): The number of characters in the chunk.
            - chunk_word_count (int): The number of words in the chunk.
            - chunk_token_count (float): An approximate number of tokens in the chunk, calculated as the length of the chunk divided by 4 (assuming 1 token ≈ 4 characters).

    Notes:
        - The function assumes that the PDF file is readable and that the `open_and_read_pdf`, `split_into_sentences`, and `chunk_sentences` functions are correctly defined.
        - The chunk's text is corrected to fix formatting issues where a period is followed directly by a capital letter without a space (e.g., "This is a sentence.Another sentence." is corrected to "This is a sentence. Another sentence.").
    """
    pages_and_texts = open_and_read_pdf(pdf_path)
    all_chunks = []
    for page in tqdm(pages_and_texts):
        sentences = split_into_sentences(page["text"], nlp)  # split sentences
        sentence_chunks = chunk_sentences(sentences=sentences, chunk_size=chunk_size)
        for chunk in sentence_chunks:
            join_chunk = (
                " ".join(chunk).replace("  ", " ").strip()
            )  # join sentences to chunk
            join_chunk = re.sub(r"\.([A-Z])", r". \1", join_chunk)  # fix format
            all_chunks.append(
                {
                    "filename": os.path.basename(pdf_path),
                    "page_number": page["page_number"],
                    "sentence_chunk": join_chunk,
                    "chunk_char_count": len(join_chunk),
                    "chunk_word_count": len(join_chunk.split(" ")),
                    "chunk_token_count": len(join_chunk) / 4,  # 1 token ~ 4 characters
                }
            )
    return all_chunks


# process all pdf files and return list of chunk
def process_multiple_pdfs(
    pdf_paths: list[str], nlp: English, chunk_size: int
) -> list[dict]:
    all_chunks = []
    for pdf_path in pdf_paths:
        chunks = process_pdf(
            pdf_path, nlp, chunk_size
        )  # process all pdf files, one by one
        all_chunks.extend(chunks)
    return all_chunks


# filter out too short sentences base on token count
def filter_chunks(chunks: list[dict], min_token_length: int) -> list[dict]:
    return [chunk for chunk in chunks if chunk["chunk_token_count"] > min_token_length]


# embedding and store into chromadb
def save_to_chromadb(
    chunks: list[dict],
    embedding_model: SentenceTransformer,
    db_name: str = "verilog_text",
    db_dir: str = "./myDB",
):
    text_chunks = [chunk["sentence_chunk"] for chunk in chunks]
    embeddings_list = (
        embedding_model.encode(text_chunks, batch_size=32, convert_to_tensor=True)
        .cpu()
        .numpy()
        .astype(float)
        .tolist()
    )
    # metadata for chunk - include text, page_number and pdf filename
    metadatas = [
        {
            "text": chunk["sentence_chunk"],
            "page_number": chunk["page_number"],
            "filename": chunk["filename"],
        }
        for chunk in chunks
    ]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # init chromadb
    # client = chromadb.Client()
    client = chromadb.PersistentClient(path=db_dir)  # pyright: ignore
    # Xóa collection cũ nếu tồn tại
    try:
        client.delete_collection(db_name)
    except ValueError:
        pass
    # Create a collection
    collection = client.get_or_create_collection(name=db_name)
    # collection.delete(where={})
    # collection.requests.delete()
    collection.add(ids=ids, embeddings=embeddings_list, metadatas=metadatas)
    print(f"✅ Data saved to: {db_dir} | Collection: '{db_name}'")


# Main execution
if __name__ == "__main__":
    # Khởi tạo spaCy để xử lý câu
    nlp = English()
    nlp.add_pipe("sentencizer")  # Thêm pipeline để chia câu

    # Xử lý tất cả các file PDF, mỗi đoạn gồm 10 câu
    chunk_size = 10
    all_chunks = process_multiple_pdfs(pdf_files, nlp, chunk_size)

    # Lọc các đoạn có số token lớn hơn 30 (đoạn quá ngắn bị loại)
    min_token_length = 30
    filtered_chunks = filter_chunks(all_chunks, min_token_length)
    embedding_model = SentenceTransformer(
        model_name_or_path="all-mpnet-base-v2",
        device=device,
    )
    save_to_chromadb(filtered_chunks, embedding_model)

    # PERF: debug point
    # Retrieve the first chunk using its ID ("chunk_0")
    # Initialize the client and retrieve the collection
    # client = chromadb.PersistentClient(path="./myDB/")  # pyright: ignore
    # collection = client.get_or_create_collection(name="verilog_text")
    # data = collection.get(include=["metadatas", "embeddings"])
    # ids = data["ids"]
    # # Randomly select 3 IDs from database
    # random_ids = random.sample(ids, 3)
    # random_results = collection.get(ids=random_ids, include=["metadatas", "embeddings"])
    # print("Random 3 values text from chromadb: ")
    # for metadata in random_results["metadatas"]:
    #     print(f'{metadata["text"]}\n')
