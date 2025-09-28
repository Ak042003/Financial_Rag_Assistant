# rag_components/generative_responder.py
from groq import Groq

class GenerativeResponder:
    """Generates a final, synthesized response and lists the sources used."""
    def __init__(self, api_key: str, model: str):
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate_response(self, query: str, context: list) -> str:
        # Format the context to be rich and readable for the LLM
        # Each context item is now a dictionary with 'document' and 'metadata'
        context_str = ""
        for i, item in enumerate(context, 1):
            source_name = "Unknown"
            doc = item.get('document', '')
            meta = item.get('metadata', {})
            
            if meta.get('source') == 'faq':
                source_name = f"FAQ: {meta.get('question', '')}"
            elif meta.get('source') == 'fund':
                source_name = f"Fund Document: {meta.get('fund_name', '')}"
            elif meta.get('source') == 'structured_query':
                source_name = "Structured Data Query"

            context_str += f"--- Source [{i}]: {source_name} ---\n{doc}\n\n"

        system_prompt = f"""
        You are a helpful financial assistant. Your task is to answer the user's query based ONLY on the provided context.

        After providing a comprehensive answer, you MUST list the sources you used in a "## Sources" section.
        Cite the sources by their number (e.g., [1], [2]).

        --- CONTEXT ---
        {context_str}
        ---

        INSTRUCTIONS:
        1.  Analyze the provided context and the user's query.
        2.  Formulate a clear, concise, and direct answer using Markdown formatting.
        3.  At the end of your response, create a "## Sources" heading and list the full names of the sources you used, citing their corresponding numbers.
        4.  Base your answer and sources strictly on the context provided. Do not make up information.

        --- EXAMPLE RESPONSE FORMAT ---
        ## Comparison of Funds

        Based on the provided data, Fund A has a higher CAGR... [Your full answer here]

        ## Sources
        * [1] Fund Document: Fund A - Direct Plan
        * [2] Fund Document: Fund B - Regular Plan
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content