"""
Motor RAG — Retrieval-Augmented Generation
LLM: Llama3-8b via Groq API (gratuita)
Autor: Eider
"""

import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

PROMPT_TEMPLATE = """Eres un asistente experto en la Organizacion Carvajal S.A.
Responde en espanol usando UNICAMENTE el contexto proporcionado.
Si no encuentras la respuesta, dilo claramente.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""

def call_llm(prompt_text, groq_token):
    headers = {
        "Authorization": f"Bearer {groq_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 512,
        "temperature": 0.3
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers, json=payload, timeout=60
    )
    result = response.json()
    if "choices" not in result:
        raise ValueError(f"Error Groq API: {result}")
    return result["choices"][0]["message"]["content"]

def load_vectorstore(vectorstore_path, hf_token="HF_TOKEN"):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        vectorstore_path, embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def build_rag_chain(groq_token, vectorstore_path="../vectorstore/carvajal_faiss"):
    vectorstore = load_vectorstore(vectorstore_path)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return {"retriever": retriever, "groq_token": groq_token}

def query(chain_dict, question):
    retriever = chain_dict["retriever"]
    groq_token = chain_dict["groq_token"]
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = call_llm(prompt, groq_token)
    sources = list(set([
        doc.metadata.get("source", "desconocido") for doc in docs
    ]))
    return answer, sources