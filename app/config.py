import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = "College AI Chatbot"
    API_VERSION: str = "v1"
    
    # Groq Configuration
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    
    # Gemini Configuration (Optional now)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"

    
    # App Settings
    APP_ENV: str = "production"
    DEBUG: bool = False
    
    # Data Settings
    KNOWLEDGE_BASE_PATH: str = "data/knowledge_base.json"
    
    # Security
    ADMIN_TOKEN: str = "admin_secret_token_123"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 20

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

settings = Settings()
