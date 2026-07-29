"""
Pipeline principal RAG — Carvajal Assistant
Ejecutar una vez para construir el vectorstore

Autor: Eider
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from scraper import build_knowledge_base
from vectorstore_builder import build_vectorstore

def main():
    print("=" * 60)
    print("RAG ASSISTANT — ORGANIZACION CARVAJAL")
    print("Construccion de base de conocimiento y vectorstore")
    print("=" * 60)

    print("\n[1/2] Recolectando documentacion de Carvajal...")
    build_knowledge_base()

    print("\n[2/2] Construyendo vectorstore FAISS...")
    build_vectorstore()

    print("\nSistema listo.")
    print("Para iniciar el chat: streamlit run dashboard.py")

if __name__ == "__main__":
    main()