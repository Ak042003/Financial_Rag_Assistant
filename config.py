# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- API and Model Settings ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANALYZER_MODEL = "openai/gpt-oss-120b"
GENERATOR_MODEL = "llama-3.1-8b-instant"
RESPONDER_MODEL = "openai/gpt-oss-120b"

# --- Database and Indexing Settings ---
CHROMA_DB_PATH = "./chroma_db"
BM25_INDEX_PATH = "bm25_searcher.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "financial_docs"

# --- Data File Paths ---
FAQ_DATA_PATH = "./data/faqs.csv"
FUNDS_DATA_PATH = "./data/funds.csv"