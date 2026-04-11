import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from duckduckgo_search import DDGS # Standard import for the library
import os
import sys

# FIX FOR LEAPCELL: Handle read-only filesystem and missing /dev/shm
# We set environment variables that tell libraries to use /tmp (writable) 
# instead of /dev/shm or the root directory.
os.environ['TMPDIR'] = '/tmp'
os.environ['PYTHON_EGG_CACHE'] = '/tmp'

# Absolute pathing ensures the 'static' folder is found regardless of how the app is started
base_dir = os.path.dirname(os.path.abspath(__file__))
static_folder_path = os.path.join(base_dir, 'static')

app = Flask(__name__, static_folder=static_folder_path, static_url_path='')
CORS(app)

# Configuration from Environment Variables
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={API_KEY}"

# Customizable Settings
MAX_SEARCH_RESULTS = 4
PRIORITY_SITES = ["pubmed.ncbi.nlm.nih.gov", "fda.gov", "ema.europa.eu", "medscape.com", "bmj.com"]

def get_optimized_search_query(user_prompt):
    """
    STAGE 1: Call Gemma to rewrite the user prompt into a professional medical search query.
    """
    print(f"--- Stage 1: Optimizing query ---")
    
    refine_prompt = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": (
                    "You are a medical search optimizer. Convert the following user prompt into a "
                    "concise, professional search query for medical databases. "
                    "Focus on drug names, interactions, or clinical conditions. "
                    "When any specific numeric patient age is mentioned (e.g., “15-year-old”, “68 y/o”, “35 years”), immediately replace it with the appropriate standard age category used in ICH pharmacovigilance guidelines (E2A, E2B(R3), E2F), ICH E7 (geriatrics), ICH E11 (pediatrics), and FDA/EMA clinical trial subpopulation reporting. Use only these conventional medical descriptors: infant, child, teenager/adolescent, young adult, middle-aged adult, older adult, elderly adult, or geriatric/very elderly patient."
                    "Mention patient's sex if given"
                    "Output ONLY the search query text, nothing else.\n\n"
                    f"User Prompt: {user_prompt}"
                )
            }]
        }],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 50}
    }

    try:
        response = requests.post(MODEL_URL, json=refine_prompt, timeout=20)
        data = response.json()
        optimized_query = data['candidates'][0]['content']['parts'][0]['text'].strip()
        return optimized_query
    except Exception as e:
        print(f"Query Optimization Error: {e}")
        return user_prompt

def get_web_context(optimized_query, max_results=4):
    """Fetches snippets and prepares HTML references with result counting."""
    search_query = optimized_query
    if PRIORITY_SITES:
        site_filter = " OR ".join([f"site:{site}" for site in PRIORITY_SITES])
        search_query = f"({site_filter}) {optimized_query}"

    context_text, formatted_refs, links = "", "", []
    
    try:
        with DDGS() as ddgs:
            # 1. Attempt priority search
            priority_results = list(ddgs.text(search_query, max_results=max_results))
            
            # 2. Attempt general search if needed
            general_results = []
            if not priority_results:
                general_results = list(ddgs.text(optimized_query, max_results=max_results))

            # DEBUG PRINTS FOR YOUR CONSOLE
            print(f"DEBUG: Optimized Query: '{optimized_query}'")
            print(f"DEBUG: Priority Site Results: {len(priority_results)}")
            print(f"DEBUG: General Search Results: {len(general_results)}")

            # Combine results for processing (logic favors priority if they exist)
            final_results = priority_results if priority_results else general_results

            if final_results:
                for i, r in enumerate(final_results, 1):
                    context_text += f"\n[SOURCE {i}]\nTitle: {r['title']}\nSnippet: {r['body']}\n"
                    formatted_refs += (
                        f"{i}. <a href='{r['href']}' target='_blank' rel='noopener noreferrer' "
                        f"style='color: #1976d2; text-decoration: underline;'>{r['title']}</a><br>\n"
                    )
                    links.append({"title": r['title'], "url": r['href']})
                
    except Exception as e:
        print(f"Search Execution Error: {e}")
        
    return context_text, formatted_refs, links

@app.route('/chat', methods=['POST'])
def chat():
    print("\n--- New Request Received ---")
    try:
        incoming = request.json
        user_history = incoming.get("contents", [])
        
        if not user_history:
            return jsonify({"error": {"message": "Empty content"}}), 400

        last_user_message = ""
        for message in reversed(user_history):
            if message.get("role") == "user":
                last_user_message = message["parts"][0]["text"]
                break
        
        # 1. STAGE 1: Optimize query
        optimized_query = get_optimized_search_query(last_user_message)
        print(f"DEBUG: The optimized query text is: {optimized_query}")

        # 2. Search & Prepare Context
        web_context, reference_html, source_links = get_web_context(optimized_query)

        # 3. STAGE 2: Final Response Generation
        SYS_TEXT = (
            "You are Alex, a Senior Pharmacovigilance MD. Provide evidence-based responses "
            "to HCPs regarding ADRs, DDIs, and safety. \n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Use the [SEARCH_DATA] block to inform your clinical reasoning.\n"
            "2. Cite sources in-text using [1], [2], etc.\n"
            "3. Section '4. References' MUST be a verbatim copy of the [REFERENCE_LIST] provided below.\n"
            "4. Structure: 1. Executive Summary; 2. Mechanism & Incidence; 3. Clinical Management; 4. References.\n"
            "5. Use CTCAE grading for severity.\n\n"
            f"[SEARCH_DATA]\n{web_context if web_context else 'No external data found.'}\n\n"
            f"[REFERENCE_LIST]\n{reference_html if reference_html else 'None available.'}\n\n"
            "End every response with: 'Clinical decisions must be tailored to the individual patient profile. Verify with current SmPC/PI.'\n"
            "If the question is not directed towards your field of pharmacovigilance expertise, state that you cannot respond."
        )

        full_contents = [
            {"role": "user", "parts": [{"text": f"SYSTEM_INSTRUCTIONS: {SYS_TEXT}"}]},
            {"role": "model", "parts": [{"text": "Understood. Analyzing search data now."}]},
            *user_history
        ]

        payload = {
            "contents": full_contents,
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 2048
            }
        }

        print("--- Stage 2: Generating final response ---")
        response = requests.post(MODEL_URL, json=payload, timeout=45)
        data = response.json()

        # --- ADDED: TOKEN COUNT PRINTING ---
        if "usageMetadata" in data:
            usage = data["usageMetadata"]
            prompt_tokens = usage.get("promptTokenCount", 0)
            total_tokens = usage.get("totalTokenCount", 0)
            print(f"DEBUG: Full Context Size (Input): {prompt_tokens} tokens")
            print(f"DEBUG: Total Transaction Size: {total_tokens} tokens")
        # ------------------------------------
        
        if "candidates" in data:
            data["web_sources"] = source_links
            data["optimized_search_query"] = optimized_query
        
        return jsonify(data)

    except Exception as e:
        print(f"Logic Error: {str(e)}")
        return jsonify({"error": {"message": str(e)}}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)