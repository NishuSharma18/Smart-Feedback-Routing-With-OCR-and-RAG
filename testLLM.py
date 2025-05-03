import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
prompt = os.environ["FEEDBACK_PROMPT"]
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

def clean_feedback(text):
    prompt_for_cleanText = prompt.replace("{text}", text)
    print(prompt_for_cleanText)
    print('\n')
    response = client.chat.completions.create(
        model="llama3.2",
        #model-"phi"
        messages=[{"role": "user", "content": prompt_for_cleanText}]
    )
    return response.choices[0].message.content.strip()

response = clean_feedback("The Marketing Team implemented dynamic pricing, increasing average order value by 10%.")
print(response)