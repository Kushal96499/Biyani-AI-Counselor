import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

# Define Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
    APP_NAME: str = "Biyani AI Counselor"
    API_VERSION: str = "v1"
    
    # Qdrant Cloud Configuration
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "biyani_clean_v2"

    # NVIDIA NIM Configuration (Priority 1)
    NVIDIA_API_KEY: Optional[str] = None

    # Groq Configuration
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    
    # Gemini Configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    # OpenRouter
    OPENROUTER_API_KEY: Optional[str] = None

    # App Settings
    APP_ENV: str = "production"
    DEBUG: bool = False
    
    # Security - MUST be set via environment variable, no insecure default
    ADMIN_TOKEN: str = "CHANGE_ME_IN_ENV"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 20

    # Caching Control
    CACHE_ENABLED: bool = False

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

settings = Settings()

# Critical startup validation
if settings.ADMIN_TOKEN == "CHANGE_ME_IN_ENV":
    import warnings
    warnings.warn(
        "⚠️  SECURITY WARNING: ADMIN_TOKEN is not set in environment variables! "
        "Set ADMIN_TOKEN in your .env file or Vercel environment before deploying.",
        stacklevel=2
    )

