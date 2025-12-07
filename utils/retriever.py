import chromadb
import numpy as np
import streamlit as st
import uuid  # pour générer des IDs uniques

class Retriever:
    def __init__(self, texts=None, embeddings=None):
        """
        texts: liste de textes/documents
        embeddings: embeddings correspondants (numpy array)
        """
        # Créer le client Chroma une seule fois par session
        if "chroma_client" in st.session_state:
            self.client = st.session_state.chroma_client
        else:
            self.client = chromadb.Client()
            st.session_state.chroma_client = self.client

        # Créer ou récupérer la collection
        self.collection_name = "rag_collection"
        existing_collections = [col.name for col in self.client.list_collections()]
        if self.collection_name in existing_collections:
            self.collection = self.client.get_collection(self.collection_name)
        else:
            self.collection = self.client.create_collection(name=self.collection_name)

        # Ajouter des documents initiaux si fournis
        if texts and embeddings:
            self.add_documents(texts, embeddings)

    def add_documents(self, texts, embeddings):
        """Ajouter des documents et leurs embeddings avec des IDs uniques"""
        for text, emb in zip(texts, embeddings):
            self.collection.add(
                ids=[str(uuid.uuid4())],  # ID unique
                documents=[text],
                embeddings=[emb.tolist()]
            )

    def query(self, query_embedding, n_results=3, threshold=0.7):
        """
        Récupère les n_results documents les plus proches.
        Si le mot exact est présent dans un document, il est toujours inclus.
        """
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )

        docs = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]

        filtered_docs = []

        # fallback exact match
        query_text = st.session_state.last_query.lower() if "last_query" in st.session_state else ""

        for doc, dist in zip(docs, distances):
            similarity = 1 - dist
            if similarity >= threshold or query_text in doc.lower():
                filtered_docs.append(doc)

        return filtered_docs
