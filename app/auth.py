from fastapi import Header, HTTPException
from app.config import settings

async def verify_admin(x_admin_token: str = Header(None)):
    """Verifies the admin token from the request header."""
    if not x_admin_token or x_admin_token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized Admin Access")
    return True
