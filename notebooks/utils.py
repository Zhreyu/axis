import os
import requests
import random
from pprint import pprint

# For reading DOCX, XLSX
import docx
import openpyxl

# -----------------------------------------------------------------------------
# 0) LOAD DOCUMENTS (Text, Word, Excel) AND COMBINE INTO SAMPLE_DOCS
# -----------------------------------------------------------------------------

def load_documents():
    """
    Load all text files from the notebooks/samplefiles folder.
    Ignores Excel and DOCX files.
    """
    documents = []
    samples_dir = "./notebooks/samplefiles"
    
    try:
        # List all files in the directory
        for filename in os.listdir(samples_dir):
            file_path = os.path.join(samples_dir, filename)
            
            # Only process text files (files ending with .txt)
            if filename.lower().endswith('.txt'):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        documents.append(content)
                except Exception as e:
                    documents.append(f"[Could not read {filename}: {e}]")
    except Exception as e:
        documents.append(f"[Error accessing directory {samples_dir}: {e}]")
    
    # Return empty list if no documents were loaded
    return documents if documents else ["No text files found in notebooks/samplefiles directory"]

# Load them into a list
SAMPLE_DOCS = load_documents()

# -----------------------------------------------------------------------------
# 1) Groq LLM API Helper
# -----------------------------------------------------------------------------
def call_groq_llm(prompt: str, model: str = "llama-3.3-70b-versatile"):
    """
    Calls the Groq LLM using the instructions you provided.

    We require an environment variable: GROQ_API_KEY
    Example usage:
      response = call_groq_llm("Explain the importance of fast language models")
      print(response)
    """
    api_key = os.environ.get("GROQ_API_KEY", "GROQ_API_KEY")
    if not api_key:
        return "[Groq LLM Error] GROQ_API_KEY not set in environment."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt
        }]
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        # The Groq API returns JSON with 'choices', 'message', etc. Let's parse it:
        result_json = resp.json()
        # print(result_json)
        # Usually the completion text is in result_json["choices"][0]["message"]["content"]
        if "choices" in result_json and len(result_json["choices"]) > 0:
            content = result_json["choices"][0]["message"]["content"]
            return f"[Groq LLM] {content}"
        else:
            return "[Groq LLM] No completion found in response."
    except requests.exceptions.RequestException as e:
        return f"[Groq LLM Error] {e}"
