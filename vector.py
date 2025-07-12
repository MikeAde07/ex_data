# logic for vectorizing our documents
# need embedding model to take text and convert it to vector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd


def process_to_csv_chroma(file, persist_directory="./chroma_db"):
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
                metadata=metadata
            )
            documents.append(document)

    return documents
