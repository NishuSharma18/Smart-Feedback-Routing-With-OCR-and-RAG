from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
import numpy as np

with open("data.json", "r") as f:
    team_docs = json.load(f)

#model = SentenceTransformer("all-MiniLM-L6-v2")
#model = SentenceTransformer("BAAI/bge-base-en-v1.5")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
corpus = list(team_docs.values())
corpus_embeddings = model.encode(corpus, convert_to_numpy=True)

index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index.add(corpus_embeddings)

faiss.write_index(index, "team_index.faiss")
with open("team_labels.pkl", "wb") as f:
    pickle.dump(list(team_docs.keys()), f)
