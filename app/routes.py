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

class AdminTextRequest(BaseModel):
    text: str
    source_name: str

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
        asyncio.create_task(cache_manager.log_chat_to_redis(user_msg, result["answer"], ip, response_time))
        
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
        # 1. Basic URL Validation
        if not url.startswith("http"):
            return HTMLResponse(content="Invalid PDF URL provided.", status_code=400)
            
        logger.info(f"Proxying PDF request for: {url}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf, */*",
        }
        
        # 2. Increased timeout and robust fetching
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch PDF from {url}. Status: {response.status_code}")
                return HTMLResponse(content=f"Error: Could not fetch PDF (Status {response.status_code})", status_code=404)
            
            # 3. Verify it's actually a PDF
            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                logger.warning(f"Resource at {url} might not be a PDF. Content-Type: {content_type}")
                # We still try to serve it, but log a warning
            
            # 4. Serve with proper headers for iframe display
            # We omit X-Frame-Options to allow framing and use CSP for modern browsers
            headers = {
                "Content-Disposition": "inline; filename=\"document.pdf\"",
                "Cache-Control": "public, max-age=3600",
                "Content-Security-Policy": "frame-ancestors 'self' *", # Allow framing
            }
            
            return StreamingResponse(
                response.aiter_bytes(), 
                media_type="application/pdf",
                headers=headers
            )
    except httpx.ConnectError:
        logger.error(f"DNS/Connection Error while fetching PDF: {url}")
        return HTMLResponse(content="Error: Could not connect to the PDF host. Please check your internet connection.", status_code=502)
    except Exception as e:
        logger.error(f"PDF Proxy Error: {str(e)}")
        return HTMLResponse(content=f"Error loading PDF: {str(e)}", status_code=500)

# --- Admin Panel Endpoints ---

@router.get("/admin/stats", dependencies=[Depends(verify_admin)])
async def get_admin_stats():
    db_stats = await rag_engine.get_collection_stats()
    redis_stats = await cache_manager.get_redis_stats()
    return {**db_stats, **redis_stats, "active_requests": active_requests}

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
        
        if await rag_engine.add_texts([text], [{"url": req.url, "title": req.url, "source_type": "Website"}]):
            await cache_manager.clear_cache()
            return {"message": "URL scraped and cache cleared successfully."}
        raise Exception("Database insertion failed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/admin/add-text", dependencies=[Depends(verify_admin)])
async def admin_add_text(req: AdminTextRequest):
    try:
        if await rag_engine.add_texts([req.text], [{"title": req.source_name, "source_type": "Manual Entry"}]):
            await cache_manager.clear_cache()
            return {"message": "Text added and cache cleared successfully."}
        raise Exception("Database insertion failed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/admin/upload-pdf", dependencies=[Depends(verify_admin)])
async def admin_upload_pdf(file: UploadFile = File(...)):
    import fitz
    try:
        content = await file.read()
        doc = fitz.open(stream=content, filetype="pdf")
        full_text = ""
        
        for page in doc:
            # 1. Attempt to extract tables as Markdown
            try:
                tabs = page.find_tables()
                if tabs.tables:
                    page_content = ""
                    last_y = 0
                    for tab in tabs:
                        # Get text before table
                        pre_table_text = page.get_text("text", clip=(0, last_y, page.rect.width, tab.bbox[1]))
                        page_content += pre_table_text + "\n"
                        
                        # Convert table to markdown
                        grid = tab.extract()
                        if grid:
                            md_table = "\n"
                            for i, row in enumerate(grid):
                                clean_row = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
                                md_table += "| " + " | ".join(clean_row) + " |\n"
                                if i == 0: # Header separator
                                    md_table += "| " + " | ".join(["---"] * len(clean_row)) + " |\n"
                            page_content += md_table + "\n"
                        last_y = tab.bbox[3]
                    
                    # Get remaining text after last table
                    post_table_text = page.get_text("text", clip=(0, last_y, page.rect.width, page.rect.height))
                    page_content += post_table_text
                    full_text += page_content + "\n\n"
                else:
                    # No tables found, use sorted text extraction
                    full_text += page.get_text("text", sort=True) + "\n\n"
            except Exception as e:
                # Fallback for older PyMuPDF versions or errors
                full_text += page.get_text("text", sort=True) + "\n\n"
        
        if await rag_engine.add_texts([full_text], [{"title": file.filename, "source_type": "PDF", "url": file.filename}]):
            await cache_manager.clear_cache()
            return {"message": f"PDF '{file.filename}' indexed with table support."}
        raise Exception("Database insertion failed.")
    except Exception as e:
        logger.error(f"PDF Upload Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/admin/clear", dependencies=[Depends(verify_admin)])
async def admin_clear_db():
    await rag_engine.clear_database()
    await cache_manager.clear_cache()
    return {"message": "Database and Redis cache cleared successfully."}

@router.get("/admin/search-chunks", dependencies=[Depends(verify_admin)])
async def admin_search_chunks(q: str):
    try:
        chunks = await rag_engine.search_chunks(q)
        return {"chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/admin/delete-chunk/{chunk_id}", dependencies=[Depends(verify_admin)])
async def admin_delete_chunk(chunk_id: str):
    try:
        success = await rag_engine.delete_chunk(chunk_id)
        if success:
            await cache_manager.clear_cache()
            return {"message": "Chunk deleted and cache cleared."}
        raise Exception("Deletion failed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/admin/boost-chunk/{chunk_id}", dependencies=[Depends(verify_admin)])
async def admin_boost_chunk(chunk_id: str):
    try:
        success = await rag_engine.boost_chunk(chunk_id)
        if success:
            await cache_manager.clear_cache()
            return {"message": "Chunk boosted! It will now appear higher in search results."}
        raise Exception("Boost failed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
