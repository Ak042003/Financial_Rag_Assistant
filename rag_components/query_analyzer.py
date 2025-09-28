# rag_components/query_analyzer.py
import json
from groq import Groq

class LLMQueryAnalyzer:
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.model = model
        self.client = Groq(api_key=api_key)

    def analyze(self, query: str, max_retries: int = 2) -> dict[str, any]:
        """
        Analyze query with LLM. Retry if JSON parsing fails.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                result = self._analyze_with_llm(query)
                if "query_type" in result and "intent" in result:
                    return result
                else:
                    raise ValueError("LLM did not return the expected JSON keys.")
            except Exception as e:
                attempt += 1
                print(f"⚠️ Groq attempt {attempt} failed: {e}")

        raise RuntimeError(f"Groq LLM failed after {max_retries} attempts for query: {query}")

    def _analyze_with_llm(self, query: str) -> dict[str, any]:
        """Send query to Groq LLM for structured analysis."""
        system_prompt = """
        You are an expert financial query analysis engine.
        Your task is to classify the user's query into one of the predefined types and state the user's intent.
        Return a structured JSON object with "query_type" and "intent".

        --- QUERY TYPE DEFINITIONS ---
        1.  "definition": The user is asking for the definition or explanation of a concept (e.g., "what is...", "explain...").
        2.  "entity_lookup": The user is asking for information about a single, specific entity like a fund.
        3.  "comparison": The user wants to compare two or more specific entities (e.g., "... vs ...", "compare...").
        4.  "ranking": The user wants a list of entities ranked by a specific metric (e.g., "top 5", "highest sharpe", "funds with CAGR > 10").
        5.  "hybrid_analytical": The user's query is a mix of a subjective quality and a quantitative filter (e.g., "safest fund with...", "best performing large cap...").
        6.  "ambiguous": The user's query is vague and lacks specific, actionable criteria (e.g., "best funds?").

        --- EXAMPLES ---
        Query: "What is CAGR?"
        JSON: {"query_type": "definition", "intent": "User wants the definition of Compound Annual Growth Rate (CAGR)."}

        Query: "Tell me about Axis Bluechip fund"
        JSON: {"query_type": "entity_lookup", "intent": "User wants to retrieve information about the Axis Bluechip fund."}

        Query: "Compare Axis Bluechip vs HDFC Top 100"
        JSON: {"query_type": "comparison", "intent": "User wants a comparison between the Axis Bluechip and HDFC Top 100 funds."}

        Query: "Top 5 funds by Sharpe ratio"
        JSON: {"query_type": "ranking", "intent": "User wants a ranked list of the top 5 funds based on their Sharpe ratio."}

        Query: "Safest fund with CAGR > 10%"
        JSON: {"query_type": "hybrid_analytical", "intent": "User wants to find a 'safe' fund that also meets the quantitative criteria of having a CAGR greater than 10%."}

        Query: "Best funds right now"
        JSON: {"query_type": "ambiguous", "intent": "User wants a subjective recommendation for the best funds to invest in, without providing specific criteria."}
        ---

        Now, analyze the following user query. IMPORTANT: Respond with only a single, valid JSON object.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)