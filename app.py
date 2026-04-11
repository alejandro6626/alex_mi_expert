import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from duckduckgo_search import DDGS
import os

# Get the absolute path to the static folder for reliability
static_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

app = Flask(__name__, static_folder=static_folder_path, static_url_path='')
CORS(app)

# Configuration from Environment Variables
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={API_KEY}"

MAX_SEARCH_RESULTS = 4
PRIORITY_SITES = ["pubmed.ncbi.nlm.nih.gov", "fda.gov", "ema.europa.eu", "medscape.com", "bmj.com"]

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

def get_optimized_search_query(user_prompt):
    if not API_KEY:
        return user_prompt
        
    refine_prompt = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": (
                    "You are a medical search optimizer. Convert the following user prompt into a "
                    "concise, professional search query for medical databases. "
                    "Focus on drug names, interactions, or clinical conditions. "
                    f"User Prompt: {user_prompt}"
                )
            }]
        }]
    }

    try:
        response = requests.post(MODEL_URL, json=refine_prompt, timeout=15)
        res_data = response.json()
        query = res_data['candidates'][0]['content']['parts'][0]['text'].strip()
        return query.replace('"', '')
    except Exception:
        return user_prompt

@app.route('/chat', methods=['POST'])
def chat():
    try:
        input_data = request.json
        user_history = input_data.get("contents", [])
        
        if not user_history:
            return jsonify({"error": "No message history provided"}), 400

        user_prompt = user_history[-1]["parts"][0]["text"]
        optimized_query = get_optimized_search_query(user_prompt)

        search_results = []
        with DDGS() as ddgs:
            site_filter = " OR ".join([f"site:{site}" for site in PRIORITY_SITES])
            full_query = f"{optimized_query} ({site_filter})"
            
            results = ddgs.text(full_query, max_results=MAX_SEARCH_RESULTS)
            for r in results:
                search_results.append(f"Source: {r['href']}\nTitle: {r['title']}\nSnippet: {r['body']}")

        context_text = "\n\n".join(search_results)

        SYS_TEXT = (
            "You are a professional Medical Affairs & Pharmacovigilance Assistant. "
            "Use the provided search data to answer the user query. "
            "Always cite your sources using [Source Name]. "
            f"\n\nSEARCH DATA:\n{context_text}"
        )

        full_contents = [
            {"role": "user", "parts": [{"text": f"SYSTEM_INSTRUCTIONS: {SYS_TEXT}"}]},
            {"role": "model", "parts": [{"text": "Understood."}]},
            *user_history
        ]

        payload = {
            "contents": full_contents,
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048}
        }

        response = requests.post(MODEL_URL, json=payload, timeout=45)
        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500

if __name__ == '__main__':
    # Default port for Leapcell is usually 8080 or 8000
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)