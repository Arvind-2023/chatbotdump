import asyncio
import logging
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
import ollama

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load FAISS index
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    logging.info("FAISS index loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load FAISS index: {e}")
    vectorstore = None

# FAISS Retriever Function (Explicit Cosine Similarity)
def faiss_retrieve(query: str, k=3):
    """
    Retrieves top-k relevant documents from FAISS using cosine similarity.
    """
    if vectorstore is None:
        logging.warning("FAISS index is missing. Returning empty results.")
        return []

    try:
        # Generate & Normalize Query Embedding
        query_embedding = embeddings.embed_query(query)
        query_embedding = np.array(query_embedding) / np.linalg.norm(query_embedding)  # L2 normalization

        # Search FAISS Index
        docs = vectorstore.similarity_search_by_vector(query_embedding, k=k)

        return [doc.page_content for doc in docs]

    except Exception as e:
        logging.error(f"Error in FAISS retrieval: {e}")
        return []

# Initialize Ollama LLM
try:
    llm = ChatOllama(model="llama3.2:3b")
    logging.info("Ollama LLM initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Ollama LLM: {e}")
    llm = None

# Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Agent Setup
from langchain.tools import Tool

# Define FAISS Retrieval Tool
faiss_tool = Tool(
    name="FAISS Retriever",
    func=faiss_retrieve,
    description="Retrieves relevant documents from FAISS based on the query."
)

agent_executor = initialize_agent(
    tools=[faiss_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# Query Complexity Analysis
def is_complex_query(query: str) -> bool:
    """
    Determines if a query is complex based on length and specific keywords.
    """
    complexity_keywords = ["explain", "summarize", "compare", "analyze", "difference"]

    return len(query.split()) > 10 or any(keyword in query.lower() for keyword in complexity_keywords)

# Single-Step RAG for Simple Queries
async def single_step_rag(query: str):
    """
    Retrieves documents and generates a response in one step.
    """
    retrieved_docs = faiss_retrieve(query, k=2)
    context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant documents found."

    prompt = f"""
You are AskRacine, the AI assistant for TechRacine Solutions. Follow these strict guidelines:

1️ **Concise & Professional**  
   - Keep responses **short and easy to read** unless a detailed explanation is necessary.  

2️ **Greeting & Conclusion**  
   - Start with a polite greeting.  
   - End with "Thanks for reaching out!" or similar.  

3️ **Uncertainty Handling**  
   - If unsure, say: "I don't have enough information."  
   - Never generate fake details.  

4️ **Clear, List-Wise Answers** *(Only if applicable)*  
   - Example:  
      *TechRacine services include:*  
     - AI-Powered Chatbots  
     - Cloud Computing  
     - Data Analytics  

5️ **Readable Formatting**  
   - Ensure answers are **well-structured and not cluttered**.  

6️ **Source Attribution (If Needed)**  
   - Example: "According to TechRacine’s official site, ..."  

7️ **Short & Focused Responses**  
   - Avoid unnecessary details.  

8️ **Provide Contact Info** *(When relevant)*  
   - "For more details, contact info@techracine.com."  

---
**Context:**  
{context}  

**Customer Question:**  
{query}  
"""

    try:
        response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        logging.error(f"Ollama LLM error: {e}")
        return "I'm currently unable to generate a response."

# Two-Step RAG for Complex Queries
async def two_step_rag(query: str):
    """
    Retrieves documents, summarizes them, and then generates a final answer.
    """
    retrieved_docs = faiss_retrieve(query, k=5)
    if not retrieved_docs:
        return "No relevant documents found."

    try:
        # Summarize Retrieved Context
        summary_prompt = f"Summarize the following for better understanding:\n{retrieved_docs}"
        summary = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": summary_prompt}])

        # Final Answer Generation
        final_prompt = f"Using this summary, answer the user's question:\n{summary['message']['content']}\nQuestion: {query}"
        response_text = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": final_prompt}])["message"]["content"]

        # Convert Markdown-style lists (- or *) into HTML lists
        html_response = response_text.replace("\n- ", "<li>").replace("\n* ", "<li>").replace("\n", "</li>\n<li>")
        if "<li>" in html_response:  # Ensure it's wrapped in <ul> if there are list items
            html_response = f"<ul><li>{html_response}</li></ul>"

        return html_response

    except Exception as e:
        logging.error(f"Ollama LLM error: {e}")
        return "I'm currently unable to generate a response."

# Dynamic RAG Selection
async def response_generator(query: str):
    """
    Selects an appropriate RAG strategy based on query complexity.
    """
    return await two_step_rag(query) if is_complex_query(query) else await single_step_rag(query)

# Streaming Response Generator (Word-by-Word)
async def stream_response(query):
    """
    Streams the response word by word.
    """
    response_text = await response_generator(query)
    
    for word in response_text.split():
        yield word + " "  # Send words one by one
        await asyncio.sleep(0.05)  # Simulate streaming delay

# -------- FASTAPI SETUP --------
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class Query(BaseModel):
    query: str

# Chat API Endpoint with Streaming
@app.post("/chat")
async def chat(query: Query):
    return StreamingResponse(stream_response(query.query), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)