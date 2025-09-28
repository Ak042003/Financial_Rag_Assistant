# rag_components/retrieval.py
from rank_bm25 import BM25Okapi
import chromadb

class BM25Search:
    """Handles lexical search using the BM25Okapi algorithm."""
    def __init__(self, documents, metadatas):
        self.documents = documents
        self.metadatas = metadatas
        tokenized_corpus = [doc.lower().split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, n_results: int = 3) -> list:
        """Performs a search and returns documents with metadata."""
        tokenized_query = query.lower().split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = doc_scores.argsort()[::-1][:n_results]
        return [{
            'document': self.documents[i],
            'metadata': self.metadatas[i]
        } for i in top_n_indices]

def initialize_chromadb(path, collection_name, embedding_function):
    """Initializes a persistent ChromaDB client and collection."""
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    return collection