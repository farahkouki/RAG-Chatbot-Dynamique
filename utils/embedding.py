from sentence_transformers import SentenceTransformer

# modèle pré-entraîné
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    return model.encode(texts, convert_to_numpy=True)
