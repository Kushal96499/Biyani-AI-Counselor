import requests
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("ADMIN_TOKEN", "Kushal123@")
URL = "http://127.0.0.1:8000/reindex"

print(f"--- Starting Reindexing at {URL} ---")
try:
    headers = {"X-Admin-Token": TOKEN}
    # Increased timeout to 30 minutes for 124+ PDFs and AI OCR
    response = requests.post(URL, headers=headers, timeout=1800)
    
    if response.status_code == 200:
        print("SUCCESS: Data Reindexed successfully with Semantic Embeddings!")
        print("Response:", response.json())
    else:
        print(f"FAILED: Status Code {response.status_code}")
        print("Error:", response.text)
except Exception as e:
    print(f"CONNECTION ERROR: Is your FastAPI server running? \n{e}")
