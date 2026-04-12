import os

import requests
from ddgs import DDGS  # Standard import for the library
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set!")

MODEL_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={API_KEY}"

# Customizable Settings
MAX_SEARCH_RESULTS = 4
PRIORITY_SITES = [
    "ncbi.nlm.nih.gov",  # This catches pubmed.ncbi..., pmc.ncbi..., and www.ncbi...
    "fda.gov",
    "ema.europa.eu",
    "medscape.com",
    "bmj.com",
]


@app.route("/")
def serve_frontend():
    return send_from_directory(".", "index.html")


def get_optimized_search_query(user_prompt):
    """
    STAGE 1: Call Gemma to rewrite the user prompt into a professional medical search query.
    """
    print(f"--- Stage 1: Optimizing query ---")

    refine_prompt = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "You are a medical search optimizer. Convert the following user prompt into a "
                            "concise, professional search query for medical databases. "
                            "Focus on drug names, interactions, or clinical conditions. "
                            "When any specific numeric patient age is mentioned (e.g., “15-year-old”, “68 y/o”, “35 years”), immediately replace it with the appropriate standard age category used in ICH pharmacovigilance guidelines (E2A, E2B(R3), E2F), ICH E7 (geriatrics), ICH E11 (pediatrics), and FDA/EMA clinical trial subpopulation reporting. Use only these conventional medical descriptors: infant, child, teenager/adolescent, young adult, middle-aged adult, older adult, elderly adult, or geriatric/very elderly patient."
                            "Mention patient's sex if given"
                            "Output ONLY the search query text, nothing else.\n\n"
                            f"User Prompt: {user_prompt}"
                        )
                    }
                ],
            }
        ],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 50},
    }

    try:
        response = requests.post(MODEL_URL, json=refine_prompt, timeout=20)
        data = response.json()
        optimized_query = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return optimized_query
    except Exception as e:
        print(f"Query Optimization Error: {e}")
        return user_prompt


def get_web_context(optimized_query, max_results=4):
    """
    Fetches snippets, filters homepages, and falls back to general search if needed.
    Returns: (context_text, formatted_refs, links, used_general_search)
    """
    context_text, formatted_refs, links = "", "", []
    filtered_results = []
    used_general_search = False

    def filter_links(raw_list):
        """Helper to remove homepages, static site info, and generic subpages."""
        valid = []
        # Define exact directory names that are never clinical articles
        STATIC_EXCLUSIONS = {
            "about",
            "help",
            "contact",
            "terms",
            "privacy",
            "faq",
            "support",
            "feedback",
            "legal",
            "home",
        }

        for r in raw_list:
            # 1. Clean the URL
            url = r["href"].lower().strip().rstrip("/")

            # 2. Extract the path
            # Split by // then take the domain part and split by / to get path
            domain_and_path = url.split("//")[-1]
            parts = domain_and_path.split("/", 1)
            path = parts[1] if len(parts) > 1 else ""

            # 3. Filtering Boolean Logic
            is_homepage = (path == "") or (
                path in ["index.html", "index.php", "default.aspx"]
            )
            is_static = path in STATIC_EXCLUSIONS
            is_generic = any(
                term in path
                for term in [
                    "search?",
                    "login",
                    "signup",
                    "about/",
                    "help/",
                    "contact-us",
                ]
            )
            is_too_short = len(path) > 0 and len(path) < 3  # Blocks /en, /v1, etc.

            # If it fails ALL exclusion tests, it's a valid article
            if (
                not is_homepage
                and not is_static
                and not is_generic
                and not is_too_short
            ):
                valid.append(r)
        return valid

    try:
        with DDGS() as ddgs:
            # --- STEP 1: Priority Site Search ---
            priority_count = 0
            if PRIORITY_SITES:
                site_filter = " OR ".join([f"site:{site}" for site in PRIORITY_SITES])
                search_query = f"({site_filter}) {optimized_query}"

                # Over-fetch to allow for filtered results
                priority_raw = list(
                    ddgs.text(search_query, max_results=max_results + 10)
                )
                filtered_results = filter_links(priority_raw)
                priority_count = len(filtered_results)
                """
                # NEW: Check if any of these "priority" results are actually general sites
                for r in filtered_results[:max_results]:
                    # If the URL doesn't contain any of our priority domains, it's a fallback result
                    if not any(domain in r['href'].lower() for domain in PRIORITY_SITES):
                        used_general_search = True
                """
            # --- STEP 2: General Search Fallback ---
            # Triggered if priority sites didn't yield enough article-level results
            general_count = 0
            if len(filtered_results) < max_results:
                general_raw = list(
                    ddgs.text(optimized_query, max_results=max_results + 10)
                )
                general_filtered = filter_links(general_raw)

                # Keep track of URLs already found to prevent duplicates
                existing_urls = {r["href"] for r in filtered_results}

                for gr in general_filtered:
                    if gr["href"] not in existing_urls:
                        filtered_results.append(gr)
                        general_count += 1
                        # used_general_search = True # Mark that we've added fallback data
                    if len(filtered_results) >= max_results:
                        break

            # --- DEBUG LOGS ---
            print(f"DEBUG: Optimized Query: '{optimized_query}'")
            print(f"DEBUG: Priority Site Results (Filtered): {priority_count}")
            print(f"DEBUG: General Search Results added: {general_count}")
            # print(f"DEBUG: Fallback flag active: {used_general_search}") ---> remove

            # --- STEP 3: Final Response Preparation ---
            # 1. Trim to the 4 results the LLM will actually see
            final_selection = filtered_results[:max_results]

            # 2. Reset and Recalculate based ONLY on these 4 links
            used_general_search = False
            general_count = 0

            for r in final_selection:
                url_low = r["href"].lower()
                is_priority = any(domain in url_low for domain in PRIORITY_SITES)

                if not is_priority:
                    used_general_search = True
                    general_count += 1
                    print(f"DEBUG: Non-priority site found in top selection: {url_low}")

            # 3. Build context strings
            for i, r in enumerate(final_selection, 1):
                context_text += (
                    f"\n[SOURCE {i}]\nTitle: {r['title']}\nSnippet: {r['body']}\n"
                )
                formatted_refs += (
                    f"{i}. <a href='{r['href']}' target='_blank' rel='noopener noreferrer' "
                    f"style='color: #1976d2; text-decoration: underline;'>{r['title']}</a><br>\n"
                )
                links.append({"title": r["title"], "url": r["href"]})

            # --- THE LOGS ---
        print(f"DEBUG: General Search Results added to context: {general_count}")
        print(f"DEBUG: Fallback flag active: {used_general_search}")

    except Exception as e:
        print(f"Search Execution Error: {e}")

    return context_text, formatted_refs, links, used_general_search


'''

def get_web_context(optimized_query, max_results=4):
    """Fetches snippets and prepares HTML references, filtering out homepages."""
    search_query = optimized_query
    if PRIORITY_SITES:
        site_filter = " OR ".join([f"site:{site}" for site in PRIORITY_SITES])
        search_query = f"({site_filter}) {optimized_query}"

    context_text, formatted_refs, links = "", "", []

    try:
        with DDGS() as ddgs:
            # We fetch more than max_results (e.g., +10) to have a "buffer"
            # for results we might filter out.
            raw_results = list(ddgs.text(search_query, max_results=max_results + 10))

            if not raw_results:
                raw_results = list(ddgs.text(optimized_query, max_results=max_results + 10))

            filtered_results = []
            for r in raw_results:
                url = r['href'].lower().rstrip('/')

                # Logic to identify a homepage:
                # 1. Split by // and take the part after the protocol.
                # 2. Check if there is anything after the first single slash.
                domain_parts = url.split("//")[-1].split("/", 1)
                path = domain_parts[1] if len(domain_parts) > 1 else ""

                # Filtering conditions:
                # - Path is empty (it's the root domain)
                # - Path is just a generic search/login page
                is_homepage = path == "" or path in ["search", "index.html", "index.php"]
                is_generic = any(term in path for term in ["/search?", "/login", "/signup", "/about"])

                if not is_homepage and not is_generic:
                    filtered_results.append(r)

                # Stop once we have reached the desired number of quality results
                if len(filtered_results) >= max_results:
                    break

            if filtered_results:
                for i, r in enumerate(filtered_results, 1):
                    context_text += f"\n[SOURCE {i}]\nTitle: {r['title']}\nSnippet: {r['body']}\n"
                    formatted_refs += (
                        f"{i}. <a href='{r['href']}' target='_blank' rel='noopener noreferrer' "
                        f"style='color: #1976d2; text-decoration: underline;'>{r['title']}</a><br>\n"
                    )
                    links.append({"title": r['title'], "url": r['href']})

    except Exception as e:
        print(f"Search Execution Error: {e}")

    return context_text, formatted_refs, links

'''
'''

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

'''


@app.route("/chat", methods=["POST"])
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
        print(f"DEBUG: User message is: {last_user_message}")
        print(f"DEBUG: The optimized query text is: {optimized_query}")

        # 2. Search & Prepare Context
        web_context, reference_html, source_links, used_fallback = get_web_context(
            optimized_query
        )
        #       web_context, reference_html, source_links = get_web_context(optimized_query)

        fallback_notice = ""
        if used_fallback:
            print(
                "DEBUG: Injecting disclaimer into System Prompt"
            )  # Log this to verify
            # We make it a direct instruction rather than a passive note
            fallback_notice = (
                "6. DATA INTEGRITY WARNING: The search results include general medical web sources. "
                "You MUST explicitly start your '1. Executive Summary' by stating: "
                "'Note: General Web search results have been considered due to lack of sufficient information in primary medical databases.'"
            )

        # 3. STAGE 2: Final Response Generation
        SYS_TEXT = (
            "You are Alex, a Senior Pharmacovigilance MD. Provide evidence-based responses "
            "to HCPs regarding ADRs, DDIs, and safety. \n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "0. Start with a clear title ALL IN CAPS that represents the query and the response formatted like **DESCRIPTIVE TITLE TEXT**.\n"
            "1. Use the [SEARCH_DATA] block to inform your clinical reasoning.\n"
            "2. Cite sources in-text using [1], [2], etc.\n"
            "3. Section '4. References' MUST be a verbatim copy of the [REFERENCE_LIST] provided below.\n"
            "4. Structure: Main Title; 1. Executive Summary; 2. Mechanism & Incidence; 3. Clinical Management; 4. References.\n"
            "5. Use CTCAE grading for severity.\n\n"
            f"{fallback_notice}"  # <--- THIS INJECTS THE DISCLAIMER
            f"[SEARCH_DATA]\n{web_context if web_context else 'No external data found.'}\n\n"
            f"[REFERENCE_LIST]\n{reference_html if reference_html else 'None available.'}\n\n"
            "End every response with: 'Clinical decisions must be tailored to the individual patient profile. Verify with current SmPC/PI.'\n"
            "IMPORTANT #1: If the question is not directed towards your field of pharmacovigilance expertise, state that you cannot respond in a simple message explaining why, don't assemble the reponse structure indicated in 4.\n"
            "IMPORTANT #2: If the question is a prank, bogus, e.g: includes non-existent or made-up drugs or commercial names or ludicrous questions, state that you cannot respond in a simple message explaining why but don't accuse the user of making things up or trying to tamper with the system, don't assemble the reponse structure indicated in 4.\n"
        )

        full_contents = [
            {"role": "user", "parts": [{"text": f"SYSTEM_INSTRUCTIONS: {SYS_TEXT}"}]},
            {
                "role": "model",
                "parts": [{"text": "Understood. Analyzing search data now."}],
            },
            *user_history,
        ]

        payload = {
            "contents": full_contents,
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
