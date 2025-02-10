import subprocess
import json
import asyncio
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import nest_asyncio
import ollama

# Initialize necessary settings
nest_asyncio.apply()
subprocess.Popen("ollama serve", shell=True)

# Qdrant Cloud Configuration
QDRANT_URL = "https://0dc82a37-4df5-4703-a8a6-15d0bcccc7ec.eu-central-1-0.aws.cloud.qdrant.io:6333"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ2NTEyOTIwfQ.ylXNcJkyhRBKtrw1Y6Cm0KPy7E7j5EpRDOf-zV1C5Fo"
COLLECTION_NAME = "techracine_chatbot"


# Initialize Qdrant Cloud Client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=API_KEY)

# Check if collection exists before creating
existing_collections = [col.name for col in qdrant_client.get_collections().collections]
# Create the collection if it doesn't exist
if COLLECTION_NAME not in existing_collections:
    print(f"Creating collection: {COLLECTION_NAME}")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
else:
    print(f"Collection '{COLLECTION_NAME}' already exists, skipping creation.")

# Set base URL for web scraping
base_url = "https://techracine.com/"
visited_urls = set()
results = []

# Define helper functions
def is_same_domain(url):
    base_domain = urlparse(base_url).netloc
    url_domain = urlparse(url).netloc
    return base_domain == url_domain

def get_links(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return [
        urljoin(base_url, a_tag['href']) for a_tag in soup.find_all('a', href=True)
        if is_same_domain(urljoin(base_url, a_tag['href'])) and urljoin(base_url, a_tag['href']) not in visited_urls
    ]

def get_title(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.title.string if soup.title else "Untitled Page"

def clean_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for nav in soup.find_all(['nav', 'header', 'footer']):
        nav.decompose()
    for tag in soup.find_all(['script', 'style']):
        tag.decompose()
    main_content = soup.find(['main', 'article']) or soup.find(class_=['content', 'main-content', 'article-content'])
    content = main_content.get_text(separator=' ', strip=True) if main_content else soup.body.get_text(separator=' ', strip=True)
    return '\n'.join(line for line in content.split('\n') if len(line.strip()) > 30)

async def scrape_pages():
    to_visit = [base_url]
    html2text = Html2TextTransformer()
    max_pages = 10

    while to_visit and len(visited_urls) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited_urls:
            continue
        visited_urls.add(current_url)
        try:
            loader = AsyncChromiumLoader([current_url], user_agent="MyAppUserAgent")
            docs = await loader.aload()
            if not docs:
                continue
            title = get_title(docs[0].page_content)
            cleaned_html = clean_content(docs[0].page_content)
            cleaned_doc = docs[0].copy()
            cleaned_doc.page_content = cleaned_html
            docs_transformed = html2text.transform_documents([cleaned_doc])
            content = docs_transformed[0].page_content
            if len(content.strip()) > 100:
                results.append({"url": current_url, "title": title, "content": content})
            to_visit.extend(get_links(docs[0].page_content))
        except Exception:
            continue

    with open('scraped_content.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results

asyncio.run(scrape_pages())

# Process documents and create embeddings
documents = [{"text": f"Title: {item['title']}\n\nContent: {item['content']}", "source": item['url']} for item in results]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len, separators=["\n\n", "\n", ". ", " ", ""])
chunks, metadatas = [], []

for doc in documents:
    texts = text_splitter.split_text(doc["text"])
    chunks.extend(texts)
    metadatas.extend([{"source": doc["source"]}] * len(texts))

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

# Store embeddings in Qdrant Cloud
qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        PointStruct(id=i, vector=embeddings.embed_documents([chunk])[0], payload=metadatas[i])
        for i, chunk in enumerate(chunks)
    ]
)

# Function to search Qdrant Cloud for relevant content
def get_relevant_context(query, k=2):
    query_vector = embeddings.embed_query(query)
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=k
    )
    return [
        {"content": hit.payload["source"], "relevance_score": hit.score}
        for hit in search_results
    ]

# Generate response using retrieved context
def response_generator(context, query):
    prompt = f"""You are an AI assistant for TechRacine Solutions company. Your role is to provide accurate and helpful responses to customer questions using only the information provided in the context. Please follow these guidelines:
    1. Start with greetings and end with thanks
    2. If you're not sure about something or if the information isn't in the context, say "I don't have enough information to answer that question"
    3. Be professional and courteous
    4. Keep responses clear and concise
    5. When appropriate, cite the source of information from the context
    6. For more information contact info@techracine.com
    Context:
    {context}
    Customer Question: {query}
    """

    response = ollama.chat(model='llama3.2:3b', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

# CLI chatbot
while True:
    query = input("\nEnter your query (type 'exit' to quit): ")
    if query.lower() in ['exit', 'quit']:
        print("Goodbye! Thank you for using the assistant.")
        break
    context_results = get_relevant_context(query, k=2)
    if not context_results:
        print("No relevant context found. Try a different query.")
        continue
    for idx, context_result in enumerate(context_results, 1):
        print(f"\nResult {idx}: {response_generator(context_result['content'], query)}")
