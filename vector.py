# logic for vectorizing our documents
# need embedding model to take text and convert it to vector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from chromadb.config import Settings
import tempfile
import os
import pandas as pd

#use the writeable directory in Render or Docker
#CHROMA_DB_DIR = "/data/chroma_db"
CHROMA_DB_DIR = "/tmp/chroma_db"

CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
    persist_directory=CHROMA_DB_DIR
)

def process_to_csv_chroma(df, persist_directory=CHROMA_DB_DIR):
    """Function to process the csv and upload the csv to the vector database"""

    #embedding model
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    #vectorstore database location
    #db_location = "/app/chroma_db"
    #ensures we don't have duplicate info being vectorized
    add_documents = not os.path.exists(persist_directory)

    #log/print statements
    print("Checking vector DB directory:", persist_directory)
    print("Exists?", os.path.exists(persist_directory))
    print("Will add documents?", add_documents)

    
    documents = []
    ids = []

    for i, row in df.iterrows():
            # Build readable row summary
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])

        metadata = {"row_index": i}
        document = Document(
            page_content=row_text,
            metadata=metadata,
            id=str(i)
        )
        documents.append(document)
        ids.append(str(i))
    
    # Add this to the vector store
    vector_store = Chroma(
        collection_name="csv_data",
        embedding_function = embedding,
        persist_directory=persist_directory,
        client_settings=CHROMA_SETTINGS
    )

    if add_documents:
        print("Number of documents prepared:", len(documents))
        print("First document:", documents[0].page_content if documents else "None")
        vector_store.add_documents(documents=documents, ids=ids)
        vector_store.persist()
        print("✅ Vector DB persisted to:", persist_directory)

#Function to create the retriever
def get_vector_retriever(persist_directory=CHROMA_DB_DIR):
    """Function to create the retriever to retrieve information from the vector database"""
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    #db_location = "/app/chroma_db"
    vector_store = Chroma(
        collection_name="csv_data",
        embedding_function = embedding,
        persist_directory=persist_directory,
        client_settings=CHROMA_SETTINGS
    )
    # look up documents and pass to prompt LLM
    retriever = vector_store.as_retriever(
        #specifies number of documents we want to look up
        search_kwargs={"k": 5}
    )
    return retriever
    