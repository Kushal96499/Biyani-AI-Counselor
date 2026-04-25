# api/config.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    GEMINI_MODEL: str = "gemini-1.5-flash"
    ADMIN_TOKEN: str = os.getenv("ADMIN_TOKEN", "Kushal123@")

settings = Settings()
