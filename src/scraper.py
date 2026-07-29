"""
Recoleccion de documentacion publica de Carvajal S.A.
para construir la base de conocimiento del RAG
Autor: Eider
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time

os.makedirs("../data", exist_ok=True)

URLS = {
    "acerca_de": "https://www.carvajal.com/inversionistas/perfil-corporativo/descripcion-del-negocio/acerca-de-nosotros/",
    "enfoque_estrategico": "https://www.carvajal.com/inversionistas/perfil-corporativo/descripcion-del-negocio/enfoque-estrategico/",
    "sostenibilidad": "https://www.carvajal.com/sostenibilidad/inicio/",
    "historia": "https://www.carvajal.com/historia/",
    "proposito": "https://www.carvajal.com/proposito-superior/",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def scrape_page(url, name):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
        content = "\n".join(lines)
        print(f"OK: {name} — {len(content)} caracteres")
        return {"source": name, "url": url, "content": content}
    except Exception as e:
        print(f"Error en {name}: {e}")
        return None

def build_knowledge_base():
    docs = []
    for name, url in URLS.items():
        doc = scrape_page(url, name)
        if doc:
            docs.append(doc)
        time.sleep(1)

    # Documento adicional manual sobre Carvajal
    manual_doc = {
        "source": "resumen_corporativo",
        "url": "manual",
        "content": """
Organizacion Carvajal — Resumen Corporativo

Carvajal S.A. es una empresa colombiana con mas de 120 anos de historia, fundada en Cali, Colombia.
Es una compania de origen familiar cuyo principal accionista es la Fundacion Carvajal.

SECTORES DE NEGOCIO:
1. Sector Papel y Empaques: Carvajal Empaques, Carvajal Pulpa y Papel, Carvajal Educacion (marca Norma)
2. Sector Tecnologia: Carvajal Tecnologia y Servicios, Carvajal Soluciones de Comunicacion, Americas BPS
3. Sector Inmobiliario: Carvajal Espacios

PRESENCIA GEOGRAFICA: 12 paises incluyendo Colombia, Mexico, Ecuador, Peru, Costa Rica, El Salvador, Panama y Estados Unidos.

VALORES CORPORATIVOS: Orientada a resultados, progresista, cercana e innovadora con sus clientes.

PROPOSITO: Impulsamos el desarrollo economico y social haciendo las cosas bien.
La transformacion y la innovacion son el motor que impulsa el desarrollo sostenible.

FUNDACION CARVAJAL: Principal accionista de Carvajal S.A., enfocada en programas sociales
de educacion, emprendimiento y desarrollo comunitario en Colombia.

MARCAS RECONOCIDAS: Norma (educacion), Propal (papel), Carpak (empaques).

HISTORIA: Fundada en 1904 en Cali por Manuel Carvajal Valencia como una imprenta.
Durante mas de un siglo ha evolucionado de empresa tipografica a grupo empresarial diversificado.
        """
    }
    docs.append(manual_doc)

    with open("../data/knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"\nBase de conocimiento guardada: {len(docs)} documentos")
    return docs

if __name__ == "__main__":
    build_knowledge_base()