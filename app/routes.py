import time
import os
from fastapi import APIRouter, HTTPException, Request, Depends, Header
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from app.rag_engine import rag_engine
from app.logger import log_chat, logger
from app.config import settings
from slowapi import Limiter
from slowapi.util import get_remote_address
from cachetools import TTLCache

limiter = Limiter(key_func=lambda r: f"{get_remote_address(r)}-{r.headers.get('user-agent', '')}")

@router.on_event("shutdown")
async def shutdown_event():
    await rag_engine.close()

# Per-session conversation history (keyed by IP)
# chat_cache intentionally removed — cached answers ignore session context

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    pdf_url: Optional[str] = None

class AdminTextRequest(BaseModel):
    text: str
    source_name: Optional[str] = "manual"

class AdminUrlRequest(BaseModel):
    url: str

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
    q_status = "NOT_INITIALIZED"
    collections = []
    test_results = "None"
    try:
        if rag_engine._qdrant:
            cols = rag_engine._qdrant.get_collections()
            collections = [c.name for c in cols.collections]
            q_status = "CONNECTED"
            
            # Test actual retrieval
            test_chunks = await rag_engine._retrieve("admission")
            test_results = f"Found {len(test_chunks)} chunks" if test_chunks else "0 chunks found"
    except Exception as e:
        q_status = f"ERROR: {str(e)}"

    return {
        "qdrant_connection": q_status,
        "available_collections": collections,
        "test_search_result": test_results,
        "rag_engine_ready": "Yes" if rag_engine._qdrant else "No",
        "env": os.getenv("APP_ENV", "production")
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
    try:
        start_time = time.time()
        user_msg = body.message.strip()

        if not user_msg:
            raise HTTPException(status_code=400, detail="Empty message")

        # Use history from frontend if available (to survive serverless reloads)
        # Use history from frontend if available
        session_id = f"{getattr(request.client, 'host', 'unknown')}-{request.headers.get('user-agent', '')}"
        
        if body.history is not None:
            history_window = [{"role": m.role, "content": m.content} for m in body.history[-6:]]
        else:
            if session_id not in chat_history:
                chat_history[session_id] = []
            history_window = chat_history[session_id][-6:]

        # Add user message to history for the current processing context
        current_history = history_window + [{"role": "user", "content": user_msg}]

        # Query the RAG engine
        result = await rag_engine.query(user_msg, history=history_window)
        
        if not result or not result.get("answer"):
            raise Exception("RAG Engine returned empty response")

        # Update persistent history ONLY if frontend didn't provide it
        if body.history is None:
            chat_history[session_id].append({"role": "user", "content": user_msg})
            chat_history[session_id].append({"role": "assistant", "content": result["answer"]})
            chat_history[session_id] = chat_history[session_id][-10:]

        response_time = time.time() - start_time
        log_chat(user_msg, result["answer"], response_time)

        return result
    except Exception as e:
        logger.error(f"CRITICAL CHAT ERROR: {str(e)}")
        # Returning error details helps us fix the 500 issue
        return {
            "answer": f"Backend Error: {str(e)}. Please check Vercel Logs.",
            "sources": [],
            "pdf_url": None
        }

from fastapi.responses import StreamingResponse
import httpx

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
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            # Stream the content back to the client
            return StreamingResponse(
                response.aiter_bytes(),
                media_type=response.headers.get("Content-Type", "application/pdf"),
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

# ── Admin API Endpoints ───────────────────────────────────────────────────────

@router.get("/admin/stats", dependencies=[Depends(verify_admin)])
async def get_admin_stats():
    return await rag_engine.get_collection_stats()

@router.get("/admin/logs", dependencies=[Depends(verify_admin)])
async def get_admin_logs():
    # Return the recent chat history across sessions
    logs = []
    for ip, history in chat_history.items():
        logs.append({"ip": ip, "messages": history})
    return {"logs": logs}

@router.post("/admin/add-text", dependencies=[Depends(verify_admin)])
async def admin_add_text(req: AdminTextRequest):
    success = await rag_engine.add_texts([req.text], [{"source": req.source_name}])
    if success:
        return {"status": "success", "message": "Text successfully embedded and added to Qdrant."}
    raise HTTPException(status_code=500, detail="Failed to add text to Qdrant.")

@router.post("/admin/scrape", dependencies=[Depends(verify_admin)])
async def admin_scrape_url(req: AdminUrlRequest):
    # Future expansion: Web scraping
    # For now, we will fetch the URL text directly
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(req.url)
            response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract text from p, h1, h2, h3, li
        texts = []
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'li']):
            t = tag.get_text(strip=True)
            if t and len(t) > 20:
                texts.append(t)
        
        full_text = "\n".join(texts)
        if not full_text:
            raise Exception("No meaningful text extracted from URL.")
            
        success = await rag_engine.add_texts([full_text], [{"source": req.url}])
        if success:
            return {"status": "success", "message": f"Successfully scraped and added {len(texts)} paragraphs from URL."}
        raise Exception("Failed to embed or save to Qdrant.")
    except Exception as e:
        logger.error(f"Scrape failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to scrape URL: {str(e)}")

@router.get("/admin/search-chunks", dependencies=[Depends(verify_admin)])
async def admin_search_chunks(q: str):
    chunks = await rag_engine.search_chunks(q)
    return {"chunks": chunks}

@router.get("/admin/debug-embedding")
async def debug_embedding(q: str = "test"):
    vec = await rag_engine._get_nvidia_vector(q)
    return {"success": vec is not None, "length": len(vec) if vec else 0, "token_set": bool(os.getenv("OPENROUTER_API_KEY"))}

@router.delete("/admin/delete-chunk/{point_id}", dependencies=[Depends(verify_admin)])
async def admin_delete_chunk(point_id: str):
    if await rag_engine.delete_chunk(point_id):
        return {"status": "success", "message": "Chunk deleted successfully."}
    raise HTTPException(status_code=500, detail="Failed to delete chunk.")

@router.post("/admin/clear", dependencies=[Depends(verify_admin)])
async def admin_clear_db():
    await rag_engine.clear_database()
    return {"status": "success", "message": "Database cleared successfully."}
