#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Check if an environment name was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <environment_name>"
    echo "Example: $0 myenv"
    exit 1
fi

# Store the environment name
ENV_NAME=$1

echo "Creating new conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.11.11 -y

echo "Activating environment: $ENV_NAME"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Installing packages from conda-forge..."
conda install -y conda-forge::sqlite
conda install -y pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -y anaconda::pandas
conda install -y conda-forge::sentence-transformers
conda install -y conda-forge::accelerate
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
conda install -y conda-forge::pymupdf
conda install -y conda-forge::spacy
conda install -y conda-forge::cupy
pip install 'spacy[transformers,cuda124]'
pip install spacy-lookups-data
conda install -y conda-forge::python-dotenv
echo "Downloading spaCy model..."
python -m spacy download en_core_web_trf
echo "Install chromadb ..."
pip install chromadb
echo "Setup complete! Activate the environment with: conda activate $ENV_NAME"
