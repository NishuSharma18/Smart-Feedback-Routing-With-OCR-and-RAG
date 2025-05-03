import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
index = faiss.read_index("team_index.faiss")

with open("team_labels.pkl", "rb") as f:
    team_labels = pickle.load(f)

with open("test_feedback_queries.json", "r") as f:
    test_data = json.load(f)

y_true = []
y_pred = []

for entry in test_data:
    label = entry['label']
    feedback = entry['feedback']
    embedding = model.encode([feedback], convert_to_numpy=True)
    D, I = index.search(embedding, k=1)
    predicted_label = team_labels[I[0][0]]

    y_true.append(label)
    y_pred.append(predicted_label)

print("\n--- Classification Report ---\n")
report = classification_report(y_true, y_pred, labels=team_labels)
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

conf_matrix = confusion_matrix(y_true, y_pred, labels=team_labels)
plt.figure(figsize=(18, 14))
sns.heatmap(conf_matrix, annot=False, xticklabels=team_labels, yticklabels=team_labels, cmap="Blues", fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Team")
plt.ylabel("Actual Team")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("confusion_matrix.png")

print("\n Evaluation complete.")
print("Report saved to: classification_report.txt")
print("Confusion matrix saved to: confusion_matrix.png")
