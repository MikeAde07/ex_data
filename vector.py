# logic for vectorizing our documents
# need embedding model to take text and convert it to vector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd


def process_to_csv_chroma(file, persist_directory="./chroma_db"):
    """Function to process the csv and upload the csv to the vector database"""
    df = pd.read_csv(file)

    #embedding model
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    #vectorstore database location
    db_location = "./chroma_db"
    #ensures we don't have duplicate info being vectorized
    add_documents = not os.path.exists(db_location)

    if add_documents :
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
        persist_directory = db_location,
        embedding_function = embedding
    )

    if add_documents:
        vector_store.add_documents(documents=documents, ids=ids)

#Function to create the retriever
def get_vector_retriever(persist_directory="./chroma_db"):
    """Function to create the retriever to retrieve information from the vector database"""
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    db_location = "./chroma_db"
    vector_store = Chroma(
        collection_name="csv_data",
        persist_directory = db_location,
        embedding_function = embedding
    )
    # look up documents and pass to prompt LLM
    retriever = vector_store.as_retriever(
        #specifies number of documents we want to look up
        search_kwargs={"k": 5}
    )
    return retriever
    
