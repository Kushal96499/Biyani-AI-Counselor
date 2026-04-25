import os
import sys

# Ensure the 'api' folder is in the path for Vercel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(title="Biyani AI Counselor API")

# Configure CORS for Production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root check
@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "Biyani AI API"}

# Include the main logic routes
app.include_router(router, prefix="/api")

# Export for Vercel
handler = app
