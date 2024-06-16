# GPT4ALL_MODEL_NAME='llama-2-7b-chat.Q4_0.gguf' 
from chromadb.config import Settings

GPT4ALL_MODEL_NAME='mistral-7b-openorca.gguf2.Q4_0.gguf' 

GPT4ALL_MODEL_FOLDER_PATH='./models'
GPT4ALL_BACKEND='llama'
GPT4ALL_ALLOW_STREAMING=True
GPT4ALL_ALLOW_DOWNLOAD=True

CHROMA_DB_DIRECTORY='db'
DOCUMENT_SOURCE_DIRECTORY='./sources/software'
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=CHROMA_DB_DIRECTORY,
    anonymized_telemetry=False
)
TARGET_SOURCE_CHUNKS=4
CHUNK_SIZE=500
CHUNK_OVERLAP=50
HIDE_SOURCE_DOCUMENTS=False

CHROMADB_NAME = "mydb"