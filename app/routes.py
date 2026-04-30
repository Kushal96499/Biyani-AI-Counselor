import os
import time
import re
import logging
import httpx
from typing import Optional, List
from fastapi import APIRouter, Request, HTTPException, Depends, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.rag_engine import rag_engine
from app.cache_manager import cache_manager
from app.logger import log_chat, get_recent_logs, logger
from app.config import settings
from app.auth import verify_admin
import asyncio

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []

class AdminUrlRequest(BaseModel):
    url: str

# Global active requests tracker
active_requests = 0

@router.post("/chat")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def chat(body: ChatRequest, request: Request):
    global active_requests
    active_requests += 1
    try:
        start_time = time.time()
        user_msg = body.message.strip()
        history = body.history
        
        result = await rag_engine.query(user_msg, history)
        
        response_time = time.time() - start_time
        ip = request.client.host
        
        # 1. Local File Logging (For backup)
        log_chat(user_msg, result["answer"], response_time)
        
        # 2. Global Redis Logging (Background task for Vercel/Global stats)
        asyncio.create_task(cache_manager.log_chat_to_redis(user_msg, result["answer"], ip))
        
        return result
    except Exception as e:
        logger.error(f"CRITICAL CHAT ERROR: {str(e)}")
        return {
            "answer": "I'm experiencing a temporary technical difficulty. Please try again in a moment.",
            "sources": [],
            "pdf_url": None
        }
    finally:
        active_requests -= 1

@router.get("/pdf-proxy")
async def pdf_proxy(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            return StreamingResponse(response.aiter_bytes(), media_type="application/pdf")
    except Exception as e:
        return HTMLResponse(content=f"Error loading PDF: {e}", status_code=404)

# --- Admin Panel Endpoints ---

@router.get("/admin/stats", dependencies=[Depends(verify_admin)])
async def get_admin_stats():
    db_stats = await rag_engine.get_collection_stats()
    redis_stats = await cache_manager.get_redis_stats()
    return {**db_stats, **redis_stats}

@router.get("/admin/logs", dependencies=[Depends(verify_admin)])
async def get_admin_logs():
    # Return Redis-based logs for global visibility
    redis_stats = await cache_manager.get_redis_stats()
    # Format Redis logs to match previous UI structure if needed, or just return raw
    formatted_logs = [{"ip": "Global", "messages": [{"role": "system", "content": log}]} for log in redis_stats["redis_logs"]]
    return {"logs": formatted_logs}

@router.post("/admin/scrape", dependencies=[Depends(verify_admin)])
async def admin_scrape_url(req: AdminUrlRequest):
    from bs4 import BeautifulSoup
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            r = await client.get(req.url)
            r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        for s in soup(['script', 'style']): s.decompose()
        text = soup.get_text(" ", strip=True)
        
        if await rag_engine.add_texts([text], [{"url": req.url}]):
            await cache_manager.clear_cache()
            return {"message": "URL scraped and cache cleared successfully."}
        raise Exception("Database insertion failed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/admin/upload-pdf", dependencies=[Depends(verify_admin)])
async def admin_upload_pdf(file: UploadFile = File(...)):
    import fitz
    try:
        content = await file.read()
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in doc: text += page.get_text()
        
        if await rag_engine.add_texts([text], [{"source": file.filename}]):
            await cache_manager.clear_cache()
            return {"message": f"PDF '{file.filename}' indexed and cache cleared."}
        raise Exception("Database insertion failed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/admin/clear", dependencies=[Depends(verify_admin)])
async def admin_clear_db():
    await rag_engine.clear_database()
    await cache_manager.clear_cache()
    return {"message": "Database and Redis cache cleared successfully."}
