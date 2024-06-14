import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from constants import *
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DB_DIRECTORY='db'
DOCUMENT_SOURCE_DIRECTORY='./sources/docs'
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=CHROMA_DB_DIRECTORY,
    anonymized_telemetry=False
)
TARGET_SOURCE_CHUNKS=4
CHUNK_SIZE=500
CHUNK_OVERLAP=50
HIDE_SOURCE_DOCUMENTS=False

client = chromadb.PersistentClient(
    path=DOCUMENT_SOURCE_DIRECTORY,
    settings=Settings(anonymized_telemetry=False)
    )

class MyKnowledgeBase:
    def __init__(self, pdf_source_folder_path: str) -> None:
        """
        Loads pdf and creates a Knowledge base using the Chroma
        vector DB.
        Args:
            pdf_source_folder_path (str): The source folder containing 
            all the pdf documents
        """
        self.pdf_source_folder_path = pdf_source_folder_path
        self.vectorstore = None

    def load_pdfs(self):
        loader = DirectoryLoader(
            self.pdf_source_folder_path,
            glob="**/*.md",
            show_progress=True
        )
        loaded_pdfs = loader.load()

        # bs_strainer = SoupStrainer(class_=("post-content", "post-title", "post-header"))
        # loader = WebBaseLoader(
        #     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        #     bs_kwargs={"parse_only": bs_strainer},
        # )
        # docs = loader.load()
        return loaded_pdfs

    def split_documents(
        self,
        loaded_docs,
    ):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunked_docs = splitter.split_documents(loaded_docs)
        return chunked_docs

    def convert_document_to_embeddings(
        self, chunked_docs, embedder
    ):
        collection = client.get_or_create_collection(name="mydb", embedding_function=embedder)
        collection.add(
            documents=[chunked_docs]
        )

        return collection

    def return_retriever_from_persistant_vector_db(
        self, embedder
    ):
        if not os.path.isdir(CHROMA_DB_DIRECTORY):
            raise NotADirectoryError(
                "Please load your vector database first."
            )
        
        return self.vectorstore.as_retriever()
    
    def initiate_document_injetion_pipeline(self):
        loaded_pdfs = self.load_pdfs()
        chunked_documents = self.split_documents(loaded_docs=loaded_pdfs)
        
        print("=> PDF loading and chunking done.")

        embeddings = HuggingFaceEmbeddings()

        self.vectorstore = Chroma.from_documents(documents=chunked_documents, embedding=embeddings)

        print("=> vector db initialised and created.")
        print("All done")