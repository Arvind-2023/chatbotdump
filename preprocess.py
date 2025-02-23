import asyncio
import logging
import aiosqlite
from sklearn.datasets import fetch_20newsgroups
import re

# Database Configuration
DB_NAME = "scraped_data.db"

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def execute_query(query, params=(), fetch=False):
    """Executes SQL queries safely."""
    async with aiosqlite.connect(DB_NAME) as db:
        if fetch:
            async with db.execute(query, params) as cursor:
                return await cursor.fetchall()
        else:
            await db.execute(query, params)
            await db.commit()

async def create_newsgroups_table():
    """Creates a separate table for the 20 Newsgroups dataset."""
    query = """
        CREATE TABLE IF NOT EXISTS newsgroups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            content TEXT
        )
    """
    await execute_query(query)
    logging.info("20 Newsgroups table initialized.")

def clean_text(text):
    """Cleans text by removing special characters, multiple spaces, and non-alphanumeric symbols."""
    text = re.sub(r"\n+", " ", text)  # Remove new lines
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)  # Remove non-alphanumeric characters except punctuation
    return text.strip()

async def store_newsgroups_data():
    """Fetches, cleans, and stores the 20 Newsgroups dataset in the database."""
    logging.info("Fetching 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    
    await create_newsgroups_table()

    logging.info("Cleaning and inserting data into the database...")
    for category, content in zip(dataset.target_names, dataset.data):
        cleaned_content = clean_text(content)
        if len(cleaned_content) > 100:  # Store only meaningful content
            query = "INSERT INTO newsgroups (category, content) VALUES (?, ?)"
            await execute_query(query, (category, cleaned_content))
    
    logging.info("20 Newsgroups dataset successfully stored.")

async def get_all_newsgroups():
    """Fetches all stored newsgroups data."""
    query = "SELECT category, content FROM newsgroups"
    return await execute_query(query, fetch=True)

if __name__ == "__main__":
    asyncio.run(store_newsgroups_data())