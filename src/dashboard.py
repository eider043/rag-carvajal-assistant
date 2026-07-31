"""
Dashboard RAG — Asistente Inteligente Carvajal S.A.
Interfaz Streamlit con identidad visual corporativa
Autor: Eider
"""

import os
import sys
import streamlit as st

@st.cache_resource
def get_rag_chain(groq_token, path, hf_token):
    from rag_engine import build_rag_chain
    return build_rag_chain(groq_token, path, hf_token)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# ── Configuracion de pagina ──────────────────────────────────────────
st.set_page_config(
    page_title="Carvajal AI Assistant",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colores institucionales Carvajal ─────────────────────────────────
# Rojo Carvajal: #E3000F | Negro: #1A1A1A | Gris: #F5F5F5
st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    .stApp { background-color: #FFFFFF; }

    .carvajal-header {
        background: linear-gradient(135deg, #1A1A1A 0%, #E3000F 100%);
        padding: 25px 30px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .carvajal-header h1 { color: white !important; font-size: 2rem; font-weight: 800; margin: 0; }
    .carvajal-header p { color: rgba(255,255,255,0.85); margin: 5px 0 0 0; font-size: 0.95rem; }

    .user-message {
        background-color: #E3000F;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        float: right;
        clear: both;
        font-size: 0.95rem;
    }
    .assistant-message {
        background-color: #1A1A1A;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        float: left;
        clear: both;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .source-badge {
        background-color: #E3000F;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        margin-right: 5px;
        display: inline-block;
    }
    .chat-container {
        background: #FAFAFA;
        border-radius: 10px;
        padding: 20px;
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
        border: 2px solid #E3000F;
        margin-bottom: 15px;
    }
    .info-card {
        background: #FAFAFA;
        border-left: 4px solid #E3000F;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: #1A1A1A;
    }
    .info-card b { color: #E3000F; }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border-top: 3px solid #E3000F;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        color: #1A1A1A;
    }
    [data-testid="stSidebar"] { background-color: #1A1A1A; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] h2 { color: #E3000F !important; font-weight: bold; }
    [data-testid="stSidebar"] h3 { color: #E3000F !important; }
    [data-testid="stSidebar"] p { color: #CCCCCC !important; }
    [data-testid="stSidebar"] li { color: #CCCCCC !important; }
    [data-testid="stSidebar"] small { color: #999999 !important; }
    [data-testid="stSidebar"] hr { border-color: #E3000F !important; opacity: 0.4; }
    [data-testid="stSidebar"] a { color: #E3000F !important; }

    .stApp h3 { color: #1A1A1A !important; }
    .stApp h4 { color: #1A1A1A !important; }
    .stApp p { color: #1A1A1A !important; }
    .stApp label { color: #1A1A1A !important; }
    .stMarkdown { color: #1A1A1A !important; }

    .stButton > button {
        background-color: #E3000F;
        color: white !important;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover { background-color: #B0000B; color: white !important; }

    div[class*="stTextInput"] input {
        border: 2px solid #E3000F;
        border-radius: 5px;
        color: #1A1A1A;
        background-color: white;
    }
    .clearfix::after { content: ""; display: table; clear: both; }
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuracion")
    st.markdown("---")

    groq_token = st.secrets["GROQ_TOKEN"]
    hf_token = st.secrets["HF_TOKEN"]

    st.markdown("---")
    st.markdown("### Modelo LLM")
    st.markdown("**llama-3.1-8b-instant**")
    st.caption("Modelo gratuito via Groq API")

    st.markdown("### Embeddings")
    st.markdown("**paraphrase-multilingual-MiniLM-L12-v2**")
    st.caption("Optimizado para espanol e ingles")

    st.markdown("---")
    st.markdown("### Fuentes de conocimiento")
    fuentes = [
        "Acerca de Carvajal",
        "Enfoque estrategico",
        "Sostenibilidad",
        "Historia corporativa",
        "Proposito superior",
        "Resumen corporativo"
    ]
    for f in fuentes:
        st.markdown(f"- {f}")

    st.markdown("---")

    if st.button("Limpiar conversacion"):
        st.session_state.messages = []
        st.session_state.rag_chain = None
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **Eider** — Data Scientist  
    [GitHub](https://github.com/eider043) | [Fiverr](https://www.fiverr.com/eiderdatadriven)
    """)


# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class='carvajal-header'>
    <h1>Carvajal AI Assistant</h1>
    <p>Asistente inteligente con tecnologia RAG sobre la Organizacion Carvajal S.A.</p>
</div>
""", unsafe_allow_html=True)

# ── KPIs ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color:#E3000F;margin:0;font-size:1.8rem'>120+</h3>
        <p style='color:#666;margin:0;font-size:0.85rem'>Anos de historia</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color:#E3000F;margin:0;font-size:1.8rem'>12</h3>
        <p style='color:#666;margin:0;font-size:0.85rem'>Paises con presencia</p>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color:#E3000F;margin:0;font-size:1.8rem'>3</h3>
        <p style='color:#666;margin:0;font-size:0.85rem'>Sectores de negocio</p>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color:#E3000F;margin:0;font-size:1.8rem'>RAG</h3>
        <p style='color:#666;margin:0;font-size:0.85rem'>Tecnologia IA</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Estado de sesion ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# ── Inicializar RAG ──────────────────────────────────────────────────
vectorstore_path = os.path.join(BASE_DIR, "vectorstore", "carvajal_faiss")

if groq_token and st.session_state.rag_chain is None:
    if os.path.exists(vectorstore_path):
        try:
            st.session_state.rag_chain = get_rag_chain(groq_token, vectorstore_path, hf_token)
            st.success("Asistente listo. Puedes comenzar a preguntar.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Vectorstore no encontrado. Ejecuta primero: python main.py")

# ── Layout principal ─────────────────────────────────────────────────
col_chat, col_info = st.columns([2, 1])

with col_chat:
    st.markdown("### Chat con el Asistente")

    # Historial del chat
    chat_html = "<div class='chat-container'>"
    if not st.session_state.messages:
        chat_html += """
        <div style='text-align:center;color:#aaa;margin-top:150px'>
            <h3 style='color:#E3000F'>Carvajal AI Assistant</h3>
            <p>Comienza a preguntar sobre Carvajal</p>
        </div>"""
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f"<div class='user-message'>{msg['content']}</div><div class='clearfix'></div>"
            else:
                sources_html = ""
                if msg.get("sources"):
                    sources_html = "<br><small>Fuentes: "
                    for s in msg["sources"]:
                        sources_html += f"<span class='source-badge'>{s}</span>"
                    sources_html += "</small>"
                chat_html += f"<div class='assistant-message'>{msg['content']}{sources_html}</div><div class='clearfix'></div>"
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input del usuario
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Escribe tu pregunta sobre Carvajal...",
            placeholder="Ej: Cuales son los sectores de negocio de Carvajal?",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Enviar")

    if submitted and user_input:
        if not groq_token:
            st.error("Ingresa tu token de Groq en el panel lateral.")
        elif st.session_state.rag_chain is None:
            st.error("El asistente no esta listo. Verifica el token y el vectorstore.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Buscando en la documentacion de Carvajal..."):
                try:
                    from rag_engine import query
                    answer, sources = query(st.session_state.rag_chain, user_input)
                    clean_answer = answer.split("Respuesta:")[-1].strip()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": clean_answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error al procesar la pregunta: {str(e)}",
                        "sources": []
                    })
            st.rerun()

with col_info:
    st.markdown("### Preguntas sugeridas")

    preguntas = [
        "Cuales son los sectores de negocio de Carvajal?",
        "En cuantos paises opera la Organizacion Carvajal?",
        "Cual es el proposito de Carvajal?",
        "Que empresas pertenecen al sector tecnologia?",
        "Cuando fue fundada Carvajal?",
        "Que es la Fundacion Carvajal?",
        "Cuales son las marcas mas conocidas de Carvajal?",
        "Que hace Carvajal Educacion?",
    ]

    for pregunta in preguntas:
        if st.button(pregunta, key=pregunta):
            if st.session_state.rag_chain:
                st.session_state.messages.append({"role": "user", "content": pregunta})
                with st.spinner("Consultando..."):
                    try:
                        from rag_engine import query
                        answer, sources = query(st.session_state.rag_chain, pregunta)
                        clean_answer = answer.split("Respuesta:")[-1].strip()
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": clean_answer,
                            "sources": sources
                        })
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {str(e)}",
                            "sources": []
                        })
                st.rerun()
            else:
                st.warning("Configura el token primero.")

    st.markdown("---")
    st.markdown("### Como funciona el RAG")
    st.markdown("""
    <div class='info-card'>
        <b>1. Ingesta</b><br>
        Se extrae documentacion publica de carvajal.com
    </div>
    <div class='info-card'>
        <b>2. Chunking</b><br>
        Los documentos se dividen en fragmentos de 500 caracteres
    </div>
    <div class='info-card'>
        <b>3. Embeddings</b><br>
        Cada fragmento se convierte en un vector numerico
    </div>
    <div class='info-card'>
        <b>4. Recuperacion</b><br>
        FAISS busca los 4 fragmentos mas relevantes
    </div>
    <div class='info-card'>
        <b>5. Generacion</b><br>
        Llama-3.1 genera la respuesta con el contexto via Groq
    </div>
    """, unsafe_allow_html=True)