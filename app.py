# app.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from rag_components.rag_system import RAGSystem
from rag_components.utils import check_and_correct_intent
from rag_components.pydantic_models import QueryResponse, QueryRequest, Source, Clarification, Disambiguation

rag_system = RAGSystem()

# ===============================================================================
# FASTAPI INTERFACE
# ===============================================================================

app = FastAPI(
    title="Financial RAG API",
    description="An API for querying financial data using a Retrieval-Augmented Generation system.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- API Endpoints ---
@app.get("/", include_in_schema=False)
async def read_index():
    return FileResponse('static/index.html')

@app.post("/query", response_model=QueryResponse, summary="Process a Financial Query")
async def process_query(request: QueryRequest):
    """
    Processes a user query through the full RAG pipeline:
    1.  **Analyze**: Classifies the query type and intent.
    2.  **Retrieve**: Fetches relevant context using the specified search mode.
    3.  **Generate**: Synthesizes a final answer based on the context.
    """
    try:
        print(f"\nProcessing query: '{request.query}'")
        # Step 1: Analyze query type
        analyzed_q = rag_system.query_analyzer.analyze(request.query)
        query_type = analyzed_q.get("query_type")
        intent = analyzed_q.get("intent", request.query)

        # Step 2: Handle Ambiguous Intent (Clarification)
        if query_type == "ambiguous":
            print("Query is ambiguous. Generating clarification question...")
            clarification_data = rag_system.clarification_generator.generate_question(request.query)
            return QueryResponse(status="clarification_needed", clarification=Clarification(**clarification_data))

        # Step 3: Handle Ambiguous Fund Names (Disambiguation)
        if query_type in ["ranking", "comparison", "hybrid_analytical"]:
            name_check = check_and_correct_intent(intent, rag_system.router.all_fund_names)
            if name_check["status"] == "disambiguation_needed":
                return QueryResponse(status="disambiguation_needed", disambiguation=Disambiguation(**name_check))
            else:
                intent = name_check["intent"] # Use the corrected intent
        
        # Step 4: If all checks pass, proceed to retrieval and generation
        print(f"All checks passed. Executing with intent: '{intent}'")
        retrieved_items = rag_system.router.retrieve(
            {"query_type": query_type, "intent": intent}, 
            request.search_mode.value
        )
        sources_for_response = [Source(**item) for item in retrieved_items]
        context_for_llm = [item['document'] for item in retrieved_items]
        
        final_answer = rag_system.responder.generate_response(
            query=request.query,
            context=retrieved_items 
        )
        return QueryResponse(status="complete", final_answer=final_answer, sources=sources_for_response)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the query.")