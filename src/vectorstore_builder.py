"""
Construccion del vectorstore FAISS con embeddings multilingues
Autor: Eider
"""

import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

os.makedirs("../vectorstore", exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_vectorstore():
    print("Cargando base de conocimiento...")
    with open("../data/knowledge_base.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    print("Dividiendo documentos en chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []
    metadatas = []
    for doc in docs:
        splits = splitter.split_text(doc["content"])
        for split in splits:
            chunks.append(split)
            metadatas.append({"source": doc["source"], "url": doc["url"]})

    print(f"Total chunks: {len(chunks)}")
    print("Generando embeddings (puede tardar unos minutos)...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    vectorstore.save_local("../vectorstore/carvajal_faiss")

    print(f"Vectorstore guardado: {len(chunks)} chunks indexados")
    return vectorstore

if __name__ == "__main__":
    build_vectorstore()