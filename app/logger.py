import logging
import os
from datetime import datetime

# Ensure logs directory exists
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Chat logging setup
chat_logger = logging.getLogger("chat_logger")
chat_logger.setLevel(logging.INFO)

chat_handler = logging.FileHandler(os.path.join(LOG_DIR, "chat.log"))
chat_formatter = logging.Formatter('%(asctime)s - %(message)s')
chat_handler.setFormatter(chat_formatter)
chat_logger.addHandler(chat_handler)

# System logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "system.log"))
    ]
)

logger = logging.getLogger("college_chatbot")

def log_chat(question: str, answer: str, response_time: float):
    """Logs chat interactions to chat.log"""
    log_entry = f"QUESTION: {question} | ANSWER: {answer} | TIME: {response_time:.4f}s"
    chat_logger.info(log_entry)
