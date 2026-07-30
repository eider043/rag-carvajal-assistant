# RAG Carvajal Assistant — IA Generativa con LangChain y Groq

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Completado-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![AI](https://img.shields.io/badge/AI-RAG-red)
![LLM](https://img.shields.io/badge/LLM-Llama--3.1-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

## Descripcion

Asistente inteligente basado en RAG (Retrieval-Augmented Generation) que responde preguntas
sobre la Organizacion Carvajal S.A. usando documentacion corporativa publica.
Implementa un pipeline completo de IA generativa con identidad visual corporativa de Carvajal.

## Arquitectura RAG

```
Documentacion Carvajal (web scraping)
|
v
Chunking (RecursiveCharacterTextSplitter, 500 chars)
|
v
Embeddings (all-MiniLM-L6-v2)
|
v
Vectorstore FAISS (busqueda por similitud)
|
v
Recuperacion Top-4 chunks relevantes
|
v
LLM Llama-3.1-8b-instant (Groq API gratuita)
|
v
Respuesta en espanol con fuentes citadas
```

## Stack tecnologico

| Componente | Tecnologia |
|---|---|
| LLM | llama-3.1-8b-instant (gratis via Groq API) |
| Embeddings | all-MiniLM-L6-v2 |
| Vectorstore | FAISS |
| Framework RAG | LangChain |
| Interfaz | Streamlit |
| Ingesta | BeautifulSoup4 + scraping web |

## Fuentes de conocimiento

- Acerca de Carvajal S.A.
- Enfoque estrategico
- Historia corporativa (120+ anos)
- Proposito superior
- Sostenibilidad
- Resumen corporativo manual

## Uso

```bash
git clone https://github.com/eider043/rag-carvajal-assistant.git
cd rag-carvajal-assistant
pip install -r requirements.txt

# Paso 1: construir base de conocimiento y vectorstore
cd src
python main.py

# Paso 2: iniciar el chat
streamlit run dashboard.py
```

Necesitas un token gratuito de Groq:
https://console.groq.com

Para deploy en Streamlit Cloud, agrega en Settings -> Secrets:
GROQ_TOKEN = "gsk_tu_token_aqui"

## Estructura

```
rag-carvajal-assistant/
├── src/
│   ├── main.py
│   ├── scraper.py
│   ├── vectorstore_builder.py
│   ├── rag_engine.py
│   └── dashboard.py
├── notebooks/
│   └── exploracion_rag.ipynb
├── data/
│   └── knowledge_base.json
├── vectorstore/
│   └── carvajal_faiss/
└── outputs/
```

## Streamlitt

Link: 


## Autor
**Eider** — Cientifico de Datos y Estadistico  
[![Fiverr](https://img.shields.io/badge/Fiverr-Contrátame-1DBF73?logo=fiverr)](https://www.fiverr.com/eiderdatadriven)