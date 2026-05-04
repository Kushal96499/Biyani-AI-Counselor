from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.routes import router, limiter
from app.config import settings
import uvicorn
import os

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.API_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
)

# Rate limiting handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Security Headers Middleware ---
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response

# CORS — allow_credentials must be False when allow_origins is wildcard (browser spec)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)

# 1. Include API routes (High Priority)
app.include_router(router, prefix="/api")

# 2. Serve Static Assets
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path = os.path.join(BASE_DIR, "frontend")
assets_path = os.path.join(frontend_path, "assets")
if os.path.exists(assets_path):
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

# 3. Favicon mapping
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = os.path.join(assets_path, "Biyani_college_logo.jpg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return {"detail": "favicon not found"}

# 4. Root Page (Lowest Priority)
@app.get("/")
async def read_index():
    index_file = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"detail": "Frontend index.html not found"}

@app.get("/admin")
async def read_admin():
    admin_file = os.path.join(frontend_path, "admin.html")
    if os.path.exists(admin_file):
        return FileResponse(admin_file)
    return {"detail": "Admin panel not found"}

@app.on_event("startup")
async def startup_event():
    pass

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
