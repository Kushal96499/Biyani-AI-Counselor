import logging
import os
from datetime import datetime

# Ensure logs directory exists
if os.environ.get("VERCEL"):
    LOG_DIR = "/tmp/logs"
else:
    # Use absolute path for local logs too to be consistent
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_DIR = os.path.join(BASE_DIR, "logs")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

# Chat logging setup
chat_logger = logging.getLogger("chat_logger")
chat_logger.setLevel(logging.INFO)

chat_handler = logging.FileHandler(os.path.join(LOG_DIR, "chat.log"), encoding='utf-8')
chat_formatter = logging.Formatter('%(asctime)s - %(message)s')
chat_handler.setFormatter(chat_formatter)
chat_logger.addHandler(chat_handler)

# System logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "system.log"), encoding='utf-8')
    ]
)

logger = logging.getLogger("college_chatbot")

def log_chat(question: str, answer: str, response_time: float):
    """Logs chat interactions to chat.log"""
    log_entry = f"QUESTION: {question} | ANSWER: {answer} | TIME: {response_time:.4f}s"
    chat_logger.info(log_entry)

def get_recent_logs(limit: int = 50):
    """Reads the last few lines from chat.log"""
    log_path = os.path.join(LOG_DIR, "chat.log")
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-limit:]
    except Exception:
        return []

def get_chat_stats():
    """Calculates total questions and unique sessions from chat.log"""
    log_path = os.path.join(LOG_DIR, "chat.log")
    if not os.path.exists(log_path):
        return {"total_questions": 0, "unique_sessions": 0}
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            questions = content.count("QUESTION:")
            # Simple heuristic for unique sessions: count unique timestamps (approx)
            # Better: count unique occurrences of "QUESTION:" entries
            return {
                "total_questions": questions,
                "unique_sessions": len(set(re.findall(r'\d{4}-\d{2}-\d{2} \d{2}', content))) # Unique hours as proxy
            }
    except Exception:
        return {"total_questions": 0, "unique_sessions": 0}
