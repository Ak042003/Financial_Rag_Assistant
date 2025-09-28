# rag_components/clarification_generator.py
import json
from groq import Groq

class ClarificationGenerator:
    """Generates clarifying questions for ambiguous user queries."""
    def __init__(self, api_key: str, model: str):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.system_prompt = """
        You are a helpful financial assistant. The user has asked an ambiguous query.
        Your goal is to ask a clarifying question to understand their specific needs.

        Based on the user's query, generate a concise, multiple-choice question.
        The options should help turn their subjective goal into an objective metric.
        Our available metrics are:
        - High Growth (cagr_3yr)
        - Low Risk (volatility)
        - Best Risk-Adjusted Return (sharpe_ratio)

        Respond with a single, valid JSON object with two keys: "question_text" and "options".
        The "options" should be a list of strings.

        --- EXAMPLE ---
        User Query: "What are the best funds to invest in?"
        Your JSON Response:
        {
            "question_text": "What do you primarily look for in the 'best' funds?",
            "options": [
                "High Growth (Top CAGR)",
                "Low Risk (Lowest Volatility)",
                "Good Balance of Risk and Return (Top Sharpe Ratio)"
            ]
        }
        """

    def generate_question(self, query: str) -> dict:
        """Generates a structured clarifying question."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"User Query: \"{query}\""}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)