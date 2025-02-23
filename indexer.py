import asyncio
import logging
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from database import get_all_web_pages, get_all_newsgroup_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def index_data():
    """
    Fetches both web-scraped content and 20 Newsgroups data from the database,
    splits text into chunks, and stores embeddings in FAISS.
    """
    logging.info("Fetching web-scraped content from the database...")
    web_results = await get_all_web_pages()

    logging.info("Fetching 20 Newsgroups data from the database...")
    newsgroup_results = await get_all_newsgroup_data()

    if not web_results and not newsgroup_results:
        logging.warning("No data found. Ensure data is stored before indexing.")
        return

    documents = []

    # Prepare web-scraped documents
    for url, title, content in web_results:
        documents.append({
            "text": f"Title: {title}\n\nContent: {content}",
            "source": url,
            "type": "web"
        })

    # Prepare newsgroups documents
    for category, document in newsgroup_results:
        documents.append({
            "text": f"Category: {category}\n\nDocument: {document}",
            "source": f"newsgroup_{category}",
            "type": "newsgroup"
        })

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks, metadatas = [], []

    for doc in documents:
        texts = text_splitter.split_text(doc["text"])
        chunks.extend(texts)
        metadatas.extend([{"source": doc["source"], "type": doc["type"]}] * len(texts))

    logging.info(f"Generated {len(chunks)} text chunks for embedding.")

    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Ensure 'chunks' is not empty before creating FAISS index
        if chunks:
            vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
            vectorstore.save_local("faiss_index")
            logging.info("FAISS index successfully saved.")
        else:
            logging.warning("No valid text chunks found for indexing.")

    except Exception as e:
        logging.error(f"Error during FAISS indexing: {e}")


if __name__ == "__main__":
    asyncio.run(index_data())