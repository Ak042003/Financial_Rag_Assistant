# build_index.py
import os
import pickle
import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# Import local modules
import config
from rag_components.retrieval import BM25Search

def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names to be valid Python identifiers."""
    cols = df.columns
    new_cols = [re.sub(r'[^A-Za-z0-9_]+', '', c).lower() for c in cols]
    df.columns = new_cols
    return df

def create_and_persist_indexes():
    """
    Loads data, creates ChromaDB and BM25 indexes, and saves them to disk.
    This is a one-time setup process.
    """
    print("--- 🚀 Starting Indexing Pipeline ---")

    # 1. Load and Prepare Data
    print("Step 1: Loading and preparing data from CSVs...")
    try:
        faqs_df = pd.read_csv(config.FAQ_DATA_PATH)
        funds_df = pd.read_csv(config.FUNDS_DATA_PATH)
        funds_df = clean_col_names(funds_df)
    except FileNotFoundError as e:
        print(f"❌ Error: Data file not found: {e}.")
        return

    documents, metadatas, ids = [], [], []

    # Process FAQs
    for _, row in faqs_df.iterrows():
        documents.append(f"Question: {row['question']} Answer: {row['answer']}")
        metadatas.append({'source': 'faq', 'question': row['question']})
        ids.append(f"faq_{row.name}")

    # Process Funds
    for _, row in funds_df.iterrows():
        description = (
            f"{row['fund_name']} is a {row['category']} fund with a 3-year CAGR of {row['cagr_3yr']}%, "
            f"a volatility of {row['volatility']}%, and a Sharpe ratio of {row['sharpe_ratio']}."
        )
        documents.append(description)
        metadata = row.to_dict()
        metadata['source'] = 'fund'
        metadatas.append(metadata)
        ids.append(row['fund_id'])
    print("Data preparation complete.")

    # 2. Create and Populate ChromaDB (Vector Index)
    print("\nStep 2: Creating and populating ChromaDB index...")
    if os.path.exists(config.CHROMA_DB_PATH):
        print(f"Warning: ChromaDB directory '{config.CHROMA_DB_PATH}' already exists. It will be overwritten.")
    
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embedding_func
    )
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    print(f"✅ ChromaDB index created with {collection.count()} documents.")

    # 3. Create and Save BM25 Index (Lexical Index)
    print("\nStep 3: Building and saving BM25 index...")
    bm25_searcher = BM25Search(documents, metadatas)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_searcher, f)
    print(f"✅ BM25 index saved to '{config.BM25_INDEX_PATH}'.")
    
    print("\n--- ✅ Indexing Pipeline Complete! ---")


if __name__ == "__main__":
    create_and_persist_indexes()