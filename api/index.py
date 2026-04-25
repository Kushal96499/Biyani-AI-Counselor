import os
import sys

# Add root directory to sys.path so 'app' package is findable
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from fastapi import FastAPI
from app.main import app as main_app

# Create a wrapper app
app = FastAPI(title="Biyani AI Counselor API Wrapper")

# Mount the main app under /api
# This maps /api/(.*) to main_app/(.*)
# So /api/chat calls main_app's /chat
app.mount("/api", main_app)

# Root check for health
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Biyani AI API Wrapper"}

# Export for Vercel
handler = app
