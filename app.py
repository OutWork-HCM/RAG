import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import faiss
import torch
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template

# Disable GPU by setting CUDA_VISIBLE_DEVICES to an empty string
os.environ["CUDA_VISIBLE_DEVICES"] = ""

torch.classes.__path__ = []  # add this line to manually set it to empty.


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore_withOpenAI(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_vectorstore_withOpenInstructor(chunks):
    """
    We using instructor-embedding.github.io
    """
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.5, "max_length": 512},
    )  # # pyright: ignore
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    # st.write(response) # PERF: debugPoint
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents")
    if user_question:
        handle_userinput(user_question)

    # PERF: debugPoint
    # st.write(user_template.replace("{{MSG}}", "Hello robot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                # get pdf text_input
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text) # PERF: debugPoint

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)  # PERF: debugPoint

                # create vector store
                vectorstore = get_vectorstore_withOpenInstructor(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
