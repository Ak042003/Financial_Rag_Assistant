# client.py
import requests
from rich.console import Console
from rich.markdown import Markdown

# The URL where your FastAPI app is running
API_URL = "http://127.0.0.1:8000/query"

# The user's query
user_query = "Compare Quant Small Cap Fund vs Nippon India Small Cap Fund"

# The data payload to send to the API
payload = {
    "query": user_query,
    "search_mode": "hybrid" # options: lexical / semantic / hybrid
}

print(f"Sending query: '{user_query}'...")

try:
    # Call your FastAPI endpoint
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

    # Extract the Markdown string from the JSON response
    data = response.json()
    markdown_response = data.get("final_answer", "No answer found.")

    # Use the 'rich' library to render the Markdown in the terminal
    console = Console()
    console.print("\n--- 🤖 API Response ---")
    console.print(Markdown(markdown_response))
    console.print("---")

except requests.exceptions.RequestException as e:
    print(f"❌ Error calling API: {e}")