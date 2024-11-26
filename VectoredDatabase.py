import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os
from ExternalData import external_data
from transformers import pipeline

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def generate_embeddings(texts):
    encoded_input = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

embedding_dimension = 384 
index = faiss.IndexFlatL2(embedding_dimension)


embedding_dimension = 384  # this is fixed for this model "all-MiniLM-L6-v2"
index = faiss.IndexFlatL2(embedding_dimension)  # Using L2 distance for similarity search

external_embeddings = generate_embeddings(external_data)
index.add(external_embeddings)
metadata = external_data

def search_similar(query, top_k=3):
    query_embedding = generate_embeddings([query])
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": metadata[idx],
            "distance": distances[0][i]
        })
    return results

query = "I really like the animations on the homepage. They make the browsing experience fun and engaging."
# query = "hr is very arrogant and rude."
# query = "Shipping has improved, but I still sometimes receive damaged products."
# query = "The new live chat feature is helpful, but the response time could be faster."
# query = "I had a payment failure while checking out. Can you make the process more seamless?"
# query = "I’ve noticed a significant delay in delivery times recently"
results = search_similar(query)

print("\nSimilarity Search Results:")
for result in results:
    print(f"Text: {result['text']}")
    print(f"Distance: {result['distance']:.4f}")
    print("-" * 50)

def prepare_for_rag(query, results):
    context = "\n".join([result["text"] for result in results])
    rag_input = f"Query: {query}\n\nContext:\n{context}"
    return rag_input

rag_input = prepare_for_rag(query, results)

print("\nRAG Input:")
print(rag_input)
print("\n")

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

team_mapping = {
    "Design Team": ["design", "layout", "animations", "branding"],
    "Backend Team": ["backend", "api", "database", "security"],
    "Frontend Team": ["frontend", "mobile app", "dark mode", "user interface"],
    "Logistics Team": ["shipping", "delivery", "inventory", "route"],
    "Payment Team": ["payment", "checkout", "fraud detection", "transaction"],
    "Marketing Team": ["marketing", "ads", "email campaign", "seo"],
    "Sales Team": ["sales", "crm", "revenue", "targets"],
    "Customer Support Team": ["support", "live chat", "tickets", "helpdesk"],
    "Human Resources Team": ["hr", "onboarding", "employee satisfaction", "wellness"],
}

def analyze_sentiment(query):
    sentiment_result = sentiment_pipeline(query)[0]
    return sentiment_result["label"], sentiment_result["score"]


def identify_team(context):
    context_lower = context.lower()
    for team, keywords in team_mapping.items():
        if any(keyword in context_lower for keyword in keywords):
            return team
    return "Unknown Team"

query_sentiment, sentiment_score = analyze_sentiment(query)
related_team = identify_team(rag_input)

print("\n")
print("\nFinal Analysis:")
print(f"Query: {query}")
print("\n")
print(f"Sentiment: {query_sentiment} (Confidence: {sentiment_score:.2f})")
print(f"Relevant Team: {related_team}")
print("\n")


