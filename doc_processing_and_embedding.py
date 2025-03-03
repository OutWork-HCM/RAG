import os
import requests
import fitz  # pymupdf, this is better than pypdf, requires pip install pymupdf
from tqdm.auto import tqdm  # for progress bars, requires pip install tqdm
from spacy.lang.en import English  # see https://spacy.io/usage
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 1 - DOWNLOAD PDF FILE
pdf_file = "human-nutrition-text.pdf"

# Download file if not exist
if not os.path.exists(pdf_file):
    print(f"File doesn't exist, downloading ........")
    # URL file
    url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    # Send a GET request to url
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a file and save the content into it
        with open(pdf_file, "wb") as file:
            file.write(response.content)
        print(f"The file has been downloaded and saved as {pdf_file}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
else:
    print(f"File {pdf_file} exists.")

# Step 2 - EXTRACT TEXT AND STATISTICS FROM EACH PDF PAGE


def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    # clean text by removing newlines and extra whitespace
    cleaned_text = text.replace(
        "\n", " "
    ).strip()  # note: this step might be different for each doc (best to experiment)
    return cleaned_text


# Open PDF and get lines/pages
# Note: focus on text, not on figures/table....
def open_and_read_pdf(pdf_file: str) -> list[dict]:
    """
    Opens a PDF file, read its content page by page, and collects statistics.

    Parameters:
        pdf_file: (str): The file path to PDF document to be opened and read.
    Return:
        list[dict]: A list of dictionaries, each contain:
            + page number
            + character count
            + word count
            + sentence count
            + token count
            + extracted text for each page
    """
    doc = fitz.open(pdf_file)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # pyright: ignore
        text = page.get_text()  # get plain text encoded as UTF-8
        text = text_formatter(text)
        pages_and_texts.append(
            {
                "page_number": page_number
                - 41,  # adjust page number since our pdf starts on page 42
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count": len(text.split(". ")),
                "page_token_count": len(text)
                / 4,  # 1 token ~ 4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                "text": text,
            }
        )

    return pages_and_texts


pages_and_texts = open_and_read_pdf(pdf_file=pdf_file)
# PERF: debug point
# print(pages_and_texts[45])
# print("\n".join(str(item) for item in random.sample(pages_and_texts, k=3)))

# Get some status on the text
# NOTE: Let's perform a rough exploratory data analysis (EDA) to get an idea of the size of the texts (e.g. character counts, word counts etc) we're working with.
# The different sizes of texts will be a good indicator into how we should split our texts.
# Many embedding models have limits on the size of texts they can ingest, for example, the sentence-transformers model all-mpnet-base-v2 has an input size of 384 tokens.
# This means that the model has been trained in ingest and turn into embeddings texts with 384 tokens (1 token ~= 4 characters ~= 0.75 words).
# Texts over 384 tokens which are encoded by this model will be auotmatically reduced to 384 tokens in length, potentially losing some information.
# Average token count per page is 287 from this below
# df = pd.DataFrame(pages_and_texts)
# stats = df.describe().round(2)
# print(stats)

# Step 3 - FURTHER TEXT PROCESSING (SPLITTING PAGES INTO SENTENCES)
# The workflow
# Ingest text -> split into groups/chunks -> embed groups/chunks -> use these embeddings
# Use spaCy to break our text into sentences - read to install: https://spacy.io/usage
nlp = English()

# Add a sentencizer pipline, see https://spacy.io/api/sentencizer/
nlp.add_pipe("sentencizer")
# Sentencizing pipeline on our pages of text.
for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    # Make sure all sentences are strings
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    # Count teh sentences
    item["page_sentences_count_spacy"] = len(item["sentences"])

# PERF: debug point
# sample_page = random.sample(pages_and_texts, k=1)[0]
# for key, value in sample_page.items():
#     print(f"{key}: {value}")
# df = pd.DataFrame(pages_and_texts)
# print(df.describe().round(2)) # --> mean(page_sentences_count_spacy) = 10

# Step 4 - CHUNKING OUR SENTENCES TOGETHER
# NOTE: Let's take a step to break down our list of sentences/text into smaller chunks.
# As you might've guessed, this process is referred to as chunking.
# Why do we do this?
#    1. Easier to manage similar sized chunks of text.
#    2. Don't overload the embedding models capacity for tokens (e.g. if an embedding model has a capacity of 384 tokens, there could be information loss if you try to embed a sequence of 400+ tokens).
#    3. Our LLM context window (the amount of tokens an LLM can take in) may be limited and requires compute power so we want to make sure we're using it as well as possible.
# Something to note is that there are many different ways emerging for creating chunks of information/text.
# For now, we're going to keep it simple and break our pages of sentences into groups of 10 (this number is arbitrary and can be changed, I just picked it because it seemed to line up well with our embedding model capacity of 384).
# On average each of our pages has 10 sentences, and an average total of 287 tokens per page. So our groups of 10 sentences will also be ~287 tokens long.
# This gives us plenty of room for the text to embedded by our all-mpnet-base-v2 model (it has a capacity of 384 tokens).

num_sentence_chunk_size = 10


# Create a function that recursively splits a list into desired sizes
def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible)
    17 sentences will be split into two lists of [[10], [7]]
    """
    return [
        input_list[i : i + slice_size] for i in range(0, len(input_list), slice_size)
    ]


# Loop through pages and split sentenses into chunks
for item in tqdm(pages_and_texts):
    item["sentences_chunks"] = split_list(
        input_list=item["sentences"], slice_size=num_sentence_chunk_size
    )
    item["num_chunks"] = len(item["sentences_chunks"])

# PERF: debug point
# sample_page = random.sample(pages_and_texts, k=1)[0]
# for key, value in sample_page.items():
#     print(f"{key}: {value}")

# Step 5 - SPLITTING EACH CHENK INTO ITS OWN ITEM
# NOTE: Create a list of dictionaries each containing a single chunk of sentences with relative information such as page number as well statistics about each chunk

# Split each chunk into its own item
pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentences_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]

        # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(
            r"\.([A-Z])", r". \1", joined_sentence_chunk
        )  # ".A" -> ". A" for any full-stop/capital letter combo
        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # Get stats about the chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len(
            [word for word in joined_sentence_chunk.split(" ")]
        )
        chunk_dict["chunk_token_count"] = (
            len(joined_sentence_chunk) / 4
        )  # 1 token = ~4 characters

        pages_and_chunks.append(chunk_dict)
# PERF: debug point
# sample_page = random.sample(pages_and_chunks, k=1)[0]
# for key, value in sample_page.items():
#     print(f"{key}: {value}")

# Get some stats about our chunks
df = pd.DataFrame(pages_and_chunks)
print(df.describe().round(2))
"""
       page_number  chunk_char_count  chunk_word_count  chunk_token_count
count      1843.00           1843.00           1843.00            1843.00
mean        583.38            734.44            112.33             183.61
std         347.79            447.54             71.22             111.89
min         -41.00             12.00              3.00               3.00
25%         280.50            315.00             44.00              78.75
50%         586.00            746.00            114.00             186.50
75%         890.00           1118.50            173.00             279.62
max        1166.00           1831.00            297.00             457.75
"""
# Show random chunks with under 30 tokens in length
min_token_length = 30
for row in df[df["chunk_token_count"] <= min_token_length].sample(5).iterrows():
    print(
        f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}'
    )

# Filter out my chunks which only include chunks over 30 tokens in length
pages_and_chunks_over_min_token_len = df[  # pyright: ignore
    df["chunk_token_count"] > min_token_length
].to_dict(orient="records")

# PERF: debug point
# print(pages_and_chunks_over_min_token_len[:2])

# Step 6 - EMBEDDING OUR TEXT CHUNKS
# We'll use `sentence-transformers` library with `all-mpnet-base-v2` model
embedding_model = SentenceTransformer(
    model_name_or_path="all-mpnet-base-v2", device=device
)

# PERF: debug point
"""
# Create a list of sentences to turn into numbers
sentences = [
    "The Sentences Transformers library provides an easy and open-source way to create embeddings.",
    "Sentences can be embedded one by one or as a list of strings.",
    "Embeddings are one of the most powerful concepts in machine learning!",
    "Learn to use embeddings well and you'll be well on your way to being an AI engineer.",
]
# Sentences are encoded/embedded by calling model.encode()
embeddings = embedding_model.encode(sentences)
embeddings_dict = dict(zip(sentences, embeddings))
# See the embeddings
for sentence, embedding in embeddings_dict.items():
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
    """

# Send teh model to GPU
embedding_model.to("cuda")

# Create embeddings one by one on the GPU
for item in tqdm(pages_and_chunks_over_min_token_len):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])


# Turn text chunks into a single list
text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]

# Embed all texts in batches
text_chunk_embeddings = embedding_model.encode(
    text_chunks, batch_size=32, convert_to_tensor=True
)

# Step 7 - SAVING EMBEDDINGS TO DATABASE FILE (CVS FILE)
text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

# Import saved file and view
text_chunks_and_embeddings_df_load = pd.read_csv(embeddings_df_save_path)
print(text_chunks_and_embeddings_df_load.head())
