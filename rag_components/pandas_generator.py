# rag_components/pandas_generator.py
import pandas as pd
import traceback
from groq import Groq

class LLMPandasQueryGenerator:
    """
    Uses an LLM to generate a pandas query string from a user's intent.
    """
    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = Groq(api_key=api_key)

    def generate_query(self, intent: str, df: pd.DataFrame, max_retries: int = 3) -> str:
        """
        Generates a pandas query by sending the DataFrame's schema and
        sample data to the LLM for context.
        """
        # Extract rich schema information and sample data from the DataFrame
        columns = df.columns.tolist()
        data_types = df.dtypes.to_string()
        sample_data = df.head(3).to_markdown()

        system_prompt = f"""
        You are an expert Python programmer specializing in the pandas library. Your task is to convert a user's natural language query about financial funds into a single, executable line of Python code that queries a pandas DataFrame named 'df'.

        --- DataFrame Schema and Sample Data ---

        **Columns:**
        {columns}

        **Data Types (dtypes):**
        {data_types}

        **Sample Rows:**
        {sample_data}

        --- INSTRUCTIONS ---
        1.  **Output ONLY the Python code.** Do not include any explanations, comments, "python", or markdown formatting.
        2.  The code must be a single line that can be passed to Python's `eval()` function.
        3.  Use the exact column names and data types from the schema provided. For columns with spaces or special characters (like '%'), use backticks in `.query()` (e.g., `df.query("`cagr_3yr` > 10")`) or bracket notation for other methods (e.g., `df['volatility']`).
        4.  For **ranking**, use `df.sort_values(...)`. For subjective terms like "top" or "best", assume descending order for high-is-good metrics (CAGR, Sharpe Ratio) and ascending for low-is-good metrics (Volatility).
        5.  For **comparison**, use `df[df['fund_name'].isin([...])]`. Extract fund names precisely from the intent, matching the format seen in the sample data.
        6.  For **hybrid/analytical** queries, use `df.query("your_condition")`. For subjective terms like "safest", translate them to a quantitative proxy (e.g., sort by lowest 'volatility').
        7.  If the query is ambiguous and cannot be converted, return the string "ERROR: Ambiguous query".

        --- EXAMPLES ---
        Intent: "User wants a ranked list of the top 5 funds based on their Sharpe ratio."
        Code: df.sort_values(by='sharpe_ratio', ascending=False).head(5)

        Intent: "User wants a comparison between the Axis Bluechip Fund and HDFC Top 100 Fund."
        Code: df[df['fund_name'].isin(['Axis Bluechip Fund', 'HDFC Top 100 Fund'])]

        Intent: "User wants to find a 'safe' fund that also meets the quantitative criteria of having a CAGR greater than 10%."
        Code: df.query("`cagr_3yr` > 10").sort_values(by='volatility (%)', ascending=True).head(5)
        ---
        Now, convert the following user intent into a single line of Python code based on the provided DataFrame schema and data.
        """
        error_context = ""
        for attempt in range(max_retries):
            # --- 2. Construct the full user message, including error context on retries ---
            user_message = f"Intent: \"{intent}\"\n{error_context}"

            print(f"\nAttempt {attempt + 1}/{max_retries} to generate code...")
            if error_context:
                print("   ↳ Retrying with error feedback.")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0 + (attempt * 0.2) # Increase temperature slightly on retries
            )
            generated_code = response.choices[0].message.content.strip().replace("`", "")
            
            # --- 3. Validate the generated code ---
            try:
                print(f"   ↳ Generated Code: {generated_code}")
                # Test the code by executing it. 'df' is available in the local scope for eval.
                eval(generated_code)
                print("✅ Code validated successfully.")
                return generated_code # Return the successfully validated code
            except Exception:
                # --- 4. If validation fails, prepare error context for the next attempt ---
                error_trace = traceback.format_exc()
                print(f"   ↳ ❌ Code failed validation: {error_trace.splitlines()[-1]}")
                error_context = f"""
                ---
                Your previous code attempt failed.
                Previous Code: {generated_code}
                Error Traceback:
                {error_trace}
                ---
                Analyze the error and the DataFrame schema. Provide a corrected, single line of Python code.
                """
        
        # --- 5. If all retries fail, return an error ---
        print("❌ Failed to generate valid code after all retries.")
        return "ERROR: LLM failed to generate a valid query."