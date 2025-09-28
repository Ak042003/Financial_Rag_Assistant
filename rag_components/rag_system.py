import config
import pickle
from rag_components.query_analyzer import LLMQueryAnalyzer
from rag_components.retrieval import BM25Search, initialize_chromadb
from rag_components.pandas_generator import LLMPandasQueryGenerator
from rag_components.generative_responder import GenerativeResponder
from rag_components.clarification_generator import ClarificationGenerator
from rag_components.router import RetrievalRouter
from chromadb.utils import embedding_functions 
import re
import pandas as pd

def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names to be valid Python identifiers."""
    cols = df.columns
    new_cols = [re.sub(r'[^A-Za-z0-9_]+', '', c).lower() for c in cols]
    df.columns = new_cols
    return df

class RAGSystem:
    """
    A singleton class to LOAD pre-built components for the RAG system.
    This runs once when the FastAPI application starts.
    """
    def __init__(self):
        print("--- 🚀 Initializing RAG System ---")
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable not set. Please create a .env file.")

        # 1. Load Pre-built Indexes from Disk
        print("Loading pre-built indexes...")
        try:
            # Load BM25 Index
            with open(config.BM25_INDEX_PATH, "rb") as f:
                self.bm25_searcher = pickle.load(f)
            print(f"✅ Loaded BM25 index from '{config.BM25_INDEX_PATH}'.")

            # Connect to ChromaDB Collection
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)
            self.chroma_collection = initialize_chromadb(config.CHROMA_DB_PATH, config.COLLECTION_NAME, embedding_func)
            print(f"✅ Connected to ChromaDB collection with {self.chroma_collection.count()} documents.")

            # Load cleaned funds data for Pandas queries
            self.funds_df = clean_col_names(pd.read_csv(config.FUNDS_DATA_PATH))
            print("✅ Loaded funds data for structured queries.")

        except FileNotFoundError as e:
            raise RuntimeError(f"❌ Index file not found: {e}. Please run 'python build_index.py' first to create the necessary indexes.")
        except Exception as e:
            raise RuntimeError(f"❌ An unexpected error occurred during initialization: {e}")


        # 2. Initialize LLM Components
        self.query_analyzer = LLMQueryAnalyzer(api_key=config.GROQ_API_KEY, model=config.ANALYZER_MODEL)
        self.pandas_generator = LLMPandasQueryGenerator(api_key=config.GROQ_API_KEY, model=config.GENERATOR_MODEL)
        self.responder = GenerativeResponder(api_key=config.GROQ_API_KEY, model=config.RESPONDER_MODEL)
        self.clarification_generator = ClarificationGenerator(api_key=config.GROQ_API_KEY, model=config.ANALYZER_MODEL)
        
        # 3. Initialize the main router
        self.router = RetrievalRouter(self.bm25_searcher, self.chroma_collection, self.funds_df, self.pandas_generator)
        print("--- ✅ RAG System Initialized Successfully ---")