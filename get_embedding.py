import os
from utils import determine_optimal_configuration
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding(cache_dir="./model"):
    """
    Returns an embedding function using the BAAI/bge-large-en-v1.5
    Args:
        cache_dir (str): path to model store
    """
    # Make sure path exist
    os.makedirs(cache_dir, exist_ok=True)
    # Get device configuration
    config = determine_optimal_configuration()
    # Init model
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={
            "device": config["device"],
            # "torch_dtype": config["dtype"],
            # "cache_dir": cache_dir,
        },
        encode_kwargs={
            "batch_size": config["batch_size"],
            "normalize_embeddings": True,
            # "show_progress_bar": True,
            "num_workers": config["num_workers"],
        },
        cache_folder=cache_dir,
    )
    return embedding_model
