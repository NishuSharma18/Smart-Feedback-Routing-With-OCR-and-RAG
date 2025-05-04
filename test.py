from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
index = faiss.read_index("team_index.faiss")
with open("team_labels.pkl", "rb") as f:
    team_labels = pickle.load(f)

def find_responsible_team(cleaned_feedback):
    embedding = model.encode([cleaned_feedback], convert_to_numpy=True)
    D, I = index.search(embedding, k=1)
    return team_labels[I[0][0]]

if __name__ == "__main__":
    feedback = "Push notifications do not open the right screens in the app."
    team = find_responsible_team(feedback)
    feedback1 = "Dark mode is inconsistent and some screens flash white."
    team1 = find_responsible_team(feedback1)
    feedback2 = "The app freezes when I try to upload a profile picture."
    team2 = find_responsible_team(feedback2)
    t3 = find_responsible_team("The site keeps going down during peak hours and loads very slowly.")
    t4 = find_responsible_team("The service becomes unusable during system updates.")
    t5 = find_responsible_team("Frequent downtime is affecting my daily operations.")
    t6 = find_responsible_team("High latency during evening hours makes the site unusable.")

    print(t3)
    print(t4)
    print(t5)
    print(t6)
