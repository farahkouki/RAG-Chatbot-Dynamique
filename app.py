import streamlit as st
from utils.embedding import embed_texts
from utils.retriever import Retriever
from streamlit_extras.let_it_rain import rain
import time

# --- CONFIG PAGE ---
st.set_page_config(
    page_title="💬 RAG Chatbot Dynamique",
    layout="centered",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# --- DARK THEME CUSTOMIZATION ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput>div>div>input {
        background-color: #1e1e2f;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #3b3b5c;
        color: #fafafa;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #56568f;
        transform: scale(1.05);
    }
    .stTextArea>div>div>textarea {
        background-color: #1e1e2f;
        color: #fafafa;
    }
    .stMarkdown p {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- INITIALISER LA SESSION ---
if "texts" not in st.session_state:
    st.session_state.texts = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# --- HEADER ---
st.title("💬 RAG Chatbot Dynamique")
st.markdown("**Ajoutez des documents et posez vos questions en temps réel**")
st.markdown("---")

# --- SECTION AJOUT DE DOCUMENT ---
st.subheader("➕ Ajouter un document")
new_text = st.text_area("Votre texte :", "", height=80)

col1, col2 = st.columns(2)
with col1:
    if st.button("Ajouter"):
        if new_text.strip():
            # Animation petite pluie pour ajout
            rain(
                emoji="📄",
                font_size=25,
                falling_speed=6,
                animation_length=1
            )
            st.session_state.texts.append(new_text)
            emb = embed_texts([new_text])
            st.session_state.embeddings.append(emb[0])
            st.session_state.retriever = Retriever(
                st.session_state.texts,
                st.session_state.embeddings
            )
            st.success("✅ Document ajouté et base mise à jour !")
        else:
            st.warning("⚠️ Le document est vide.")

with col2:
    if st.button("Supprimer tous les documents"):
        st.session_state.texts = []
        st.session_state.embeddings = []
        st.session_state.retriever = None
        st.info("🗑️ Tous les documents ont été supprimés.")

# --- AFFICHER DOCUMENTS EXISTANTS ---
if st.session_state.texts:
    st.markdown("**📚 Documents dans la base :**")
    for i, doc in enumerate(st.session_state.texts, 1):
        st.markdown(f"{i}. {doc}")

st.markdown("---")

# --- SECTION QUESTION ---
st.subheader("❓ Posez une question")
user_input = st.text_input("Votre question :")

if user_input:
    st.session_state.last_query = user_input

    if st.session_state.retriever:
        query_emb = embed_texts([user_input])
        results = st.session_state.retriever.query(
            query_emb,
            n_results=5,
            threshold=0.6
        )

        if results:
            st.markdown("**🔍 Documents pertinents :**")
            for i, doc in enumerate(results, 1):
                st.markdown(f"✨ **{i}.** {doc}")
        else:
            st.info("❌ Aucun document pertinent trouvé.")
    else:
        st.warning("⚠️ Aucun document dans la base. Ajoutez d'abord des textes !")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#7c7c7c;font-size:12px;'>RAG Chatbot Dynamique • 2025 • Powered by Streamlit & ChromaDB</p>",
    unsafe_allow_html=True
)
