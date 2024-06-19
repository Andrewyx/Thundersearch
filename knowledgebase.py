import os
from typing import Optional

import chromadb
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from constants import *
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

client = chromadb.PersistentClient(
    path=CHROMA_DB_DIRECTORY,
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
        loader = GenericLoader.from_filesystem(
            self.pdf_source_folder_path,
            glob="**/*",
            suffixes=[".h", ".md"],
            show_progress=True,
            parser=LanguageParser()
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
        collection = client.get_or_create_collection(name=CHROMADB_NAME, embedding_function=embedder)
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
        print("=> File loading and chunking done.")

        embeddings = HuggingFaceEmbeddings()

        try:
            print("=> Loading DB")
            self.vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIRECTORY,
                client=client,
                embedding_function=embeddings
                )
        except Exception as e:
            print(e)
            print("=> DB not found, creating from new")
            
            self.vectorstore = Chroma.from_documents(
                documents=chunked_documents, 
                embedding=embeddings, 
                persist_directory=CHROMA_DB_DIRECTORY,
                client=client
                )

        print("=> vector db initialised and created.")
        print("All done")