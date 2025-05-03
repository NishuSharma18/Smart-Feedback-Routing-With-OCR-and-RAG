import cv2
import numpy as np
import pytesseract
from openai import OpenAI
from dotenv import load_dotenv
import os

def get_grayscale(image):
   return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
   return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def noise_removal(image):
   kernel = np.ones((1,1),np.uint8)
   image = cv2.dilate(image, kernel, iterations = 1) # very thin letter
   kernel = np.ones((1,1),np.uint8)
   image = cv2.erode(image, kernel, iterations = 1)  #very thick letter
   image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
   return image

def get_Text(filepath):
   img = cv2.imread(filepath)
   if img is None:
      return "Failed to load image"

   img = get_grayscale(img)
   img = thresholding(img)
   img = noise_removal(img)
   custom_config = r'-l eng+hin --oem 3 --psm 6'
   text = 'NO TEXT TO BE APPEARED'
   try:
      text = pytesseract.image_to_string(img,config=custom_config, timeout=5)
   except RuntimeError as timeout_error:
      print("OCR timed out")

   return text

load_dotenv()
prompt = os.environ["FEEDBACK_PROMPT"]
# Make sure Ollama is running
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

def clean_feedback(text):
    prompt_for_cleanText = prompt.replace("{text}", text)
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt_for_cleanText}]
    )
    return response.choices[0].message.content.strip()

from transformers import pipeline


sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
def get_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']

from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("team_index.faiss")
with open("team_labels.pkl", "rb") as f:
    team_labels = pickle.load(f)

def find_responsible_team(cleaned_feedback):
    embedding = model.encode([cleaned_feedback], convert_to_numpy=True)
    D, I = index.search(embedding, k=1)
    return team_labels[I[0][0]]


if __name__ == "__main__":
    path = r"C:\Users\asnis\Downloads\MajorProject8thSem\Screenshot 2025-05-02 024140.png"
    feedback = get_Text(path)
    feedback = clean_feedback(feedback)
    print(feedback)
    sentiment, score = get_sentiment(feedback)
    print(sentiment)
    print(score)
    team = find_responsible_team(feedback)
    print(team)