import requests
from bs4 import BeautifulSoup
from app.logger import logger
from app.utils import clean_text
import os

def scrape_urls(url_file: str):
    data = []
    if not os.path.exists(url_file):
        return data

    with open(url_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in urls:
        try:
            logger.info(f"Scraping URL: {url}")
            # Increased timeout to 30 seconds for slow college websites
            response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                text = soup.get_text()
                cleaned_text = clean_text(text)
                if cleaned_text:
                    data.append({"text": cleaned_text, "source": url})
                    logger.info(f"Successfully scraped: {url}")
            else:
                logger.error(f"Failed to scrape {url}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            
    return data
