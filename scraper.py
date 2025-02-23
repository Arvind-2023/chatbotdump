import asyncio
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncChromiumLoader
from database import create_tables, insert_web_page, get_all_web_pages  # Fixed function name

# Base URL for web scraping (Change this as needed)

BASE_URL = "https://techracine.com/"
VISITED_URLS = set()
MAX_PAGES = 10  # Set a limit on the number of pages to scrape

# Configure logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def is_same_domain(url):
    """Checks whether a given URL belongs to the same domain as the base URL."""
    return urlparse(BASE_URL).netloc == urlparse(url).netloc

def extract_links(html_content):
    """Extracts all valid same-domain links from a web page."""
    soup = BeautifulSoup(html_content, 'html.parser')
    links = [
        urljoin(BASE_URL, a_tag['href']) 
        for a_tag in soup.find_all('a', href=True)
        if is_same_domain(urljoin(BASE_URL, a_tag['href'])) and urljoin(BASE_URL, a_tag['href']) not in VISITED_URLS
    ]
    return links

def extract_title(html_content):
    """Extracts and returns the page title, or 'Untitled Page' if not available."""
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.title.string.strip() if soup.title else "Untitled Page"

def clean_text_content(html_content):
    """Removes unnecessary HTML elements and extracts meaningful text."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove non-essential tags like navigation, scripts, and styles
    for tag in soup.find_all(['nav', 'header', 'footer', 'script', 'style']):
        tag.decompose()
    
    # Extract clean text while filtering out very short lines
    return '\n'.join(line for line in soup.get_text(separator=' ', strip=True).split('\n') if len(line.strip()) > 30)

async def scrape_website():
    """Scrapes web pages, stores relevant content in the database, and returns stored pages."""
    await create_tables()

    # Check if data is already available in the database
    stored_pages = await get_all_web_pages()  # Fixed function name
    if stored_pages:
        logging.info("Data already exists in the database. Skipping web scraping.")
        return [{"url": url, "title": title, "content": content} for url, title, content in stored_pages]

    logging.info("Starting web scraping process...")
    pages_to_scrape = [BASE_URL]

    while pages_to_scrape and len(VISITED_URLS) < MAX_PAGES:
        current_url = pages_to_scrape.pop(0)

        if current_url in VISITED_URLS:
            continue

        VISITED_URLS.add(current_url)

        try:
            logging.info(f"Fetching: {current_url}")
            loader = AsyncChromiumLoader([current_url])
            docs = await loader.aload()
            
            if not docs:
                logging.warning(f"No content found at {current_url}. Skipping...")
                continue

            # Extract title and clean content
            page_title = extract_title(docs[0].page_content)
            cleaned_content = clean_text_content(docs[0].page_content)

            # Store only if the content is meaningful
            if len(cleaned_content.strip()) > 100:
                await insert_web_page(current_url, page_title, cleaned_content)  # Corrected function call
                pages_to_scrape.extend(extract_links(docs[0].page_content))  # Queue new links
            else:
                logging.warning(f"Skipping {current_url} due to insufficient content.")

        except Exception as error:
            logging.error(f"Error while processing {current_url}: {error}")

    logging.info("Web scraping process completed.")
    return await get_all_web_pages()  # Fixed function name

if __name__ == "__main__":
    asyncio.run(scrape_website())