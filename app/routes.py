import time
import os
from fastapi import APIRouter, HTTPException, Request, Depends, Header
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from app.rag_engine import rag_engine
from app.pdf_loader import load_pdfs_from_links
from app.scraper import scrape_urls
from app.logger import log_chat, logger
from app.config import settings
from slowapi import Limiter
from slowapi.util import get_remote_address
from cachetools import TTLCache

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Per-session conversation history (keyed by IP)
# chat_cache intentionally removed — cached answers ignore session context

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    pdf_url: Optional[str] = None

class HealthResponse(BaseModel):
    status: str

def verify_admin(x_admin_token: Optional[str] = Header(None)):
    if x_admin_token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "ok"}

@router.get("/debug")
async def debug_status():
    return {
        "groq_key": "Present" if settings.GROQ_API_KEY else "MISSING",
        "gemini_key": "Present" if settings.GEMINI_API_KEY else "MISSING",
        "nvidia_key": "Present" if settings.NVIDIA_API_KEY else "MISSING",
        "qdrant_url": "Present" if settings.QDRANT_URL else "MISSING",
        "rag_engine_status": "Ready" if rag_engine._qdrant and rag_engine._embedder else "NOT_INITIALIZED",
        "env": os.getenv("APP_ENV", "development")
    }

@router.get("/welcome")
async def welcome():
    return {
        "message": "Dear student, Welcome to Biyani AI Assistant. This AI chatbot has been developed after 20 years of teaching experience of team. It will help you with correct answers for all the complex questions of academic subjects. However, consult your teacher before any final decision. -Team Biyani."
    }

# In-memory history (last 5 messages per session)
chat_history = {}

@router.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def chat(request: Request, body: ChatRequest):
    start_time = time.time()
    user_msg = body.message.strip()

    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    # Session tracking by IP (simple, stateless-friendly)
    session_id = getattr(request.client, "host", "unknown")
    if session_id not in chat_history:
        chat_history[session_id] = []

    # Pass last 3 turns (6 messages) as history context
    history_window = chat_history[session_id][-6:]

    # Query the RAG engine
    result = rag_engine.query(user_msg, history=history_window)

    # Append this turn to session history
    chat_history[session_id].append({"role": "user",      "content": user_msg})
    chat_history[session_id].append({"role": "assistant", "content": result["answer"]})
    # Keep last 10 messages (~5 turns) to bound memory
    chat_history[session_id] = chat_history[session_id][-10:]

    response_time = time.time() - start_time
    log_chat(user_msg, result["answer"], response_time)

    return result

from fastapi.responses import StreamingResponse
import requests

@router.get("/pdf-proxy")
async def pdf_proxy(url: str):
    """
    Proxies external PDF requests to bypass CSP/CORS framing restrictions.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.biyanicolleges.org/",
            "Origin": "https://www.biyanicolleges.org"
        }
        response = requests.get(url, stream=True, timeout=20, headers=headers)
        response.raise_for_status()
        
        # Determine media type (PDF or HTML for flipbooks)
        media_type = "application/pdf" if ".pdf" in url.lower() else "text/html"
        
        # Stream the content back to the client
        return StreamingResponse(
            response.iter_content(chunk_size=8192),
            media_type=media_type,
            headers={
                "Content-Disposition": "inline",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except Exception as e:
        logger.error(f"PDF Proxy failed for {url}: {str(e)}")
        # Return a friendly HTML error instead of JSON for the iframe
        return HTMLResponse(
            content=f"""
            <div style="font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: #666; text-align: center; padding: 20px;">
                <svg width="64" height="64" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                <h3 style="margin-top: 15px;">Document Not Available</h3>
                <p style="font-size: 14px;">The link provided by the official website seems to be broken (404 Not Found).</p>
                <a href="{url}" target="_blank" style="margin-top: 10px; color: #8b0000; font-weight: bold; text-decoration: none;">Try direct link &rarr;</a>
            </div>
            """,
            status_code=200 # Return 200 so the iframe displays the content
        )

@router.post("/reindex", dependencies=[Depends(verify_admin)])
async def reindex(full: bool = False):
    """
    Reloads data. If full=True, clears database first. Otherwise, only adds new data.
    """
    try:
        logger.info(f"Starting reindexing (Full={full})...")
        
        if full:
            rag_engine.clear_database()
            indexed_sources = set()
        else:
            indexed_sources = rag_engine.get_indexed_sources()
            logger.info(f"Skipping {len(indexed_sources)} already indexed sources.")
        
        # 1. Load Remote PDFs from all_pdf_links.txt
        pdf_links_path = os.path.join(settings.DATA_DIR, "all_pdf_links.txt")
        if os.path.exists(pdf_links_path):
            with open(pdf_links_path, "r") as f:
                all_links = [line.strip() for line in f if line.strip()]
            
            # Filter only new links
            new_links = [l for l in all_links if l not in indexed_sources]
            if new_links:
                logger.info(f"Indexing {len(new_links)} new remote PDFs...")
                pdf_data = load_pdfs_from_links(new_links)
                rag_engine.add_documents(pdf_data)
            else:
                logger.info("No new remote PDFs to index.")

        # 2. Load Scraped Web Content
        urls_path = os.path.join(settings.DATA_DIR, "urls.txt")
        if os.path.exists(urls_path):
            with open(urls_path, "r") as f:
                all_urls = [line.strip() for line in f if line.strip()]
            
            new_urls = [u for u in all_urls if u not in indexed_sources]
            if new_urls:
                logger.info(f"Indexing {len(new_urls)} new web URLs...")
                # We need to temporarily write these new URLs to a file for scrape_urls
                # Or modify scrape_urls to accept a list. Let's keep it simple.
                temp_file = os.path.join(settings.DATA_DIR, "temp_new_urls.txt")
                with open(temp_file, "w") as f:
                    for u in new_urls: f.write(u + "\n")
                
                web_data = scrape_urls(temp_file)
                rag_engine.add_documents(web_data)
                if os.path.exists(temp_file): os.remove(temp_file)
            else:
                logger.info("No new web URLs to index.")
        
        # 3. Load FAQs (Always refresh FAQs or skip if you prefer)
        rag_engine.add_faqs(os.path.join(settings.DATA_DIR, "faqs.json"))
        
        return {"status": "success", "message": f"Reindexing completed. Added new data while keeping existing index."}
    except Exception as e:
        logger.error(f"Reindexing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")
