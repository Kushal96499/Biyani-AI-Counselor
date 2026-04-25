import os
import requests
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from api.rag_engine import LiteRAGEngine
from api.pdf_loader import load_pdfs_from_links
from api.logger import logger
from cachetools import TTLCache

router = APIRouter()

# Initialize RAG Engine with absolute path for Vercel
# Vercel functions run in a 'current directory' that might not be the root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "knowledge_base.json")
rag_engine = LiteRAGEngine(storage_path=DATA_PATH)

# Use a memory cache (Bypassed for testing as per your request earlier)
chat_cache = TTLCache(maxsize=100, ttl=3600)

def verify_admin(x_admin_token: str = Header(None)):
    admin_token = os.getenv("ADMIN_TOKEN", "Kushal123@")
    if x_admin_token != admin_token:
        raise HTTPException(status_code=403, detail="Invalid Admin Token")
    return True

@router.post("/chat")
async def chat(request: dict):
    user_message = request.get("message", "")
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message")
    
    # Bypassing cache for real-time logic changes
    response_data = rag_engine.chat(user_message)
    return response_data

@router.get("/pdf-proxy")
async def pdf_proxy(url: str = Query(...)):
    """
    Proxies PDF content to bypass CSP 'frame-ancestors' or CORS.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.biyanicolleges.org/",
            "Origin": "https://www.biyanicolleges.org"
        }
        resp = requests.get(url, headers=headers, stream=True, timeout=20)
        
        content_type = resp.headers.get("Content-Type", "application/pdf")
        
        return StreamingResponse(
            resp.iter_content(chunk_size=1024),
            media_type=content_type,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Content-Disposition": f"inline; filename=document"
            }
        )
    except Exception as e:
        logger.error(f"Proxy error for {url}: {e}")
        raise HTTPException(status_code=500, detail="Failed to proxy document")

@router.post("/reindex", dependencies=[Depends(verify_admin)])
async def reindex(full: bool = False):
    try:
        # Load links from root/data
        links_path = os.path.join(BASE_DIR, "data", "all_pdf_links.txt")
        if os.path.exists(links_path):
            with open(links_path, "r") as f:
                all_links = [line.strip() for line in f if line.strip()]
            pdf_data = load_pdfs_from_links(all_links)
            rag_engine.add_documents(pdf_data)
        return {"status": "success", "message": "Knowledge base updated"}
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        return {"status": "error", "message": str(e)}
