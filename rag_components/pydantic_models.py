from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Any, Optional

# --- Pydantic Models for API ---
class SearchMode(str, Enum):
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    LEXICAL = "lexical"

class QueryRequest(BaseModel):
    query: str
    search_mode: SearchMode = SearchMode.HYBRID

class Source(BaseModel):
    document: str
    metadata: Dict[str, Any]

class Clarification(BaseModel):
    question_text: str
    options: List[str]

class Disambiguation(BaseModel):
    question_text: str
    original_name: str
    options: List[str]

class QueryResponse(BaseModel):
    final_answer: Optional[str] = None
    sources: Optional[List[Source]] = None
    clarification: Optional[Clarification] = None
    disambiguation: Optional[Disambiguation] = None
    status: str # e.g., "complete" or "clarification_needed"