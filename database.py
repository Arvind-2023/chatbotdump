import aiosqlite
import logging

DB_NAME = "company_knowledge_base.db"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def execute_query(query, params=(), fetch=False):
    """Executes SQL queries with optional data fetching."""
    try:
        async with aiosqlite.connect(DB_NAME) as db:
            if fetch:
                async with db.execute(query, params) as cursor:
                    return await cursor.fetchall()
            else:
                await db.execute(query, params)
                await db.commit()
    except Exception as e:
        logging.error(f"Database error: {e}")
        raise

async def create_tables():
    """Creates necessary database tables if they do not exist."""
    queries = [
        """
        CREATE TABLE IF NOT EXISTS company_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            document TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS web_scraped (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            last_scraped TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]
    
    for query in queries:
        await execute_query(query)
    
    logging.info("Database tables initialized successfully.")

async def insert_company_data(category, document):
    """Inserts company-related documents into the database."""
    query = "INSERT INTO company_data (category, document) VALUES (?, ?)"
    await execute_query(query, (category, document))
    logging.info(f"Company data added: {category}")

async def insert_web_page(url, title, content):
    """Inserts or updates web-scraped pages to avoid duplicates."""
    query = """
    INSERT INTO web_scraped (url, title, content) 
    VALUES (?, ?, ?)
    ON CONFLICT(url) DO UPDATE SET 
        title = excluded.title,
        content = excluded.content,
        last_scraped = CURRENT_TIMESTAMP
    """
    await execute_query(query, (url, title, content))
    logging.info(f"Web page stored/updated: {url}")

async def get_all_company_data():
    """Retrieves all stored company data."""
    query = "SELECT category, document FROM company_data"
    return await execute_query(query, fetch=True)

async def get_all_web_pages():
    """Retrieves all web-scraped content from the database."""
    query = "SELECT url, title, content FROM web_scraped"
    return await execute_query(query, fetch=True)

# Alias for backward compatibility
async def get_all_newsgroup_data():
    return await get_all_company_data()
