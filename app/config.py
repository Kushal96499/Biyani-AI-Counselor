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
    QDRANT_COLLECTION: str = "biyani_ai_clean_v2"

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
    
    # Security
    ADMIN_TOKEN: str = "admin_secret_token_123"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 20

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

settings = Settings()
