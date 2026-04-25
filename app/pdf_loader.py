import os
import time
import requests
from pypdf import PdfReader
from app.logger import logger
from app.config import settings

def load_pdfs_from_links(pdf_links: list):
    data = []
    for url in pdf_links:
        if not url.strip(): continue
        filename = url.split('/')[-1]
        try:
            logger.info(f"Downloading and processing PDF: {filename}...")
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Failed to download {url} (Status: {response.status_code})")
                continue
                
            pdf_bytes = response.content
            full_text = ""
            
            # 1. Try PyPDF
            try:
                from io import BytesIO
                reader = PdfReader(BytesIO(pdf_bytes))
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt: full_text += txt + "\n"
            except: pass
            
            # 2. Try pdfplumber
            if not full_text.strip():
                try:
                    import pdfplumber
                    from io import BytesIO
                    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                        for page in pdf.pages:
                            txt = page.extract_text()
                            if txt: full_text += txt + "\n"
                except: pass
                
            # 3. Gemini AI OCR Fallback (with Smart Wait & Retry)
            if not full_text.strip() and settings.GEMINI_API_KEY:
                logger.info(f"Local extraction failed for {filename}. Using Gemini AI OCR...")
                
                ocr_models = [settings.GEMINI_MODEL or "gemini-2.5-flash", "gemini-1.5-flash"]
                success = False
                
                for model in ocr_models:
                    if success: break
                    
                    retries = 0
                    while retries < 3: # Try up to 3 times per model
                        try:
                            import base64
                            import json
                            gemini_url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={settings.GEMINI_API_KEY}"
                            pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
                            
                            payload = {"contents": [{"parts": [
                                {"text": "Extract all text from this PDF."},
                                {"inline_data": {"mime_type": "application/pdf", "data": pdf_b64}}
                            ]}]}
                            
                            res = requests.post(gemini_url, json=payload, timeout=90)
                            
                            if res.status_code == 200:
                                full_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                                logger.info(f"AI OCR success using {model}")
                                success = True
                                break
                            elif res.status_code == 429:
                                # Parse exact wait time from Google's response
                                try:
                                    # Default to 30s, but try to find better
                                    wait_time = 30
                                    error_json = res.json()
                                    msg = error_json.get("error", {}).get("message", "")
                                    if "retry in" in msg:
                                        import re
                                        matches = re.findall(r"retry in ([\d\.]+)s", msg)
                                        if matches: wait_time = int(float(matches[0])) + 2
                                except: wait_time = 30
                                
                                logger.warning(f"Quota hit for {model}. Waiting {wait_time}s to reset...")
                                time.sleep(wait_time)
                                retries += 1
                                continue
                            else:
                                logger.error(f"Gemini API Error {res.status_code}: {res.text}")
                                break # Try next model
                                
                        except Exception as e:
                            logger.error(f"OCR Exception: {e}")
                            break

            if full_text.strip():
                data.append({"text": full_text, "source": url})
                logger.info(f"Successfully indexed {filename}")
                time.sleep(1) # Tiny safety gap
            else:
                logger.warning(f"Failed to extract text from {filename} after all attempts.")
                
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            
    return data
