"""
rag.py — Supabase pgvector edition (v2 — enhanced retrieval & generation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Improvements:
  • Python-side in-memory exact cosine similarity search fallback (highly robust)
  • Query expansion for common short follow-up questions
  • Dynamic prompt instruction for direct recommendations with clear winners
  • Removal of PDF links from text answers for web chat, programmatic attachment for WhatsApp
"""

from __future__ import annotations

import os
import re
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from supabase import create_client
from typing import Optional

load_dotenv(override=True)

# ── Config ────────────────────────────────────────────────────────────────────
TOP_K        = 8
EMBED_MODEL  = "all-mpnet-base-v2"    # 768-dim — far better than MiniLM for insurance
MIN_SIM      = 0.20                    # minimum similarity threshold
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# ── Clients ───────────────────────────────────────────────────────────────────
print("Loading embedding model (all-mpnet-base-v2) …")
model  = SentenceTransformer(EMBED_MODEL, device="cpu")
client = Groq(api_key=GROQ_API_KEY)

if not SUPABASE_URL or not SUPABASE
    raise EnvironmentError(
        "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env"
    )
db = create_client(SUPABASE_URL, SUPABASE_KEY)
print("RAG ready! (Supabase pgvector v2 — 768-dim)")

# ── Known policy list (for "list all policies" queries) ──────────────────────
POLICY_SUMMARIES = """
Star Health Insurance offers the following policies:

1. **Arogya Sanjeevani** — IRDAI standardised plan, ₹5L–₹2Cr, 5% co-pay, cumulative bonus up to 50%.
2. **Family Health Optima (FHO)** — Family floater, ₹5L–₹25L, automatic restoration, newborn cover from day 1, loyalty bonus up to 100%.
3. **Medi Classic (Individual)** — Individual plan, ₹5L–₹15L, no co-pay, pre/post hospitalisation, long-term discounts.
4. **Star Health Assure** — Comprehensive floater for up to 9 members, unlimited restoration, wellness discount up to 20%.
5. **Star Health Premier** — Designed for 50+ age group, no upper age limit, home care, AYUSH, wellness program.
6. **Young Star Insurance** — For young adults & families, unlimited restoration, wellness rewards, Silver & Gold variants.
7. **Super Star** — Star Health's flagship plan, ₹5L–₹5Cr, no co-pay, broadest coverage.
""".strip()

_SUPABASE_STORAGE = "https://efsgbittghkwjoklhqfk.supabase.co/storage/v1/object/public/policy-pdfs"

POLICY_PDF_MAP = {
    "arogya sanjeevani": f"{_SUPABASE_STORAGE}/arogya-sanjeevani.pdf",
    "family health optima": f"{_SUPABASE_STORAGE}/family-health-optima.pdf",
    "medi classic": f"{_SUPABASE_STORAGE}/medi-classic.pdf",
    "star assure": f"{_SUPABASE_STORAGE}/star-assure.pdf",
    "star premier": f"{_SUPABASE_STORAGE}/star-premier.pdf",
    "young star": f"{_SUPABASE_STORAGE}/young-star.pdf",
    "super star": f"{_SUPABASE_STORAGE}/super-star.pdf",
}

POLICY_KEYWORDS = {
    "arogya sanjeevani": "Arogya Sanjeevani",
    "family health optima": "Family Health Optima",
    "fho": "Family Health Optima",
    "medi classic": "Medi Classic (Individual)",
    "star assure": "Star Health Assure",
    "assure": "Star Health Assure",
    "star premier": "Star Health Premier",
    "premier": "Star Health Premier",
    "young star": "Young Star Insurance",
    "super star": "Super Star",
}

LIST_PATTERNS = [
    r"(all|list|types?|kinds?|available|show|tell me|what are).*(polic|plan|insurance|cover)",
    r"(polic|plan|insurance).*(all|list|types?|available|different)",
    r"what (plans?|policies|insurance).*(offer|have|available|star health)",
    r"^(plans?|policies|options?)$",
]


def _is_list_query(question: str) -> bool:
    """Detect broad "list all policies" type queries."""
    q = question.lower().strip()
    for pat in LIST_PATTERNS:
        if re.search(pat, q):
            return True
    return False


def _extract_policy_keyword(question: str) -> Optional[str]:
    """Return policy name if the question explicitly mentions one."""
    q = question.lower()
    for keyword, name in POLICY_KEYWORDS.items():
        if keyword in q:
            return name
    return None


# ── Query Expansion ───────────────────────────────────────────────────────────
def _expand_query(query: str) -> str:
    """
    Expand vague follow-up queries with descriptive keywords to improve RAG vector retrieval.
    """
    q = query.lower()
    expansions = []

    # 1. Family-related follow-ups
    family_keywords = ["family", "wife", "husband", "children", "kids", "parents", "floater"]
    if any(k in q for k in family_keywords):
        expansions.append("family floater health insurance plan")

    # 2. Senior citizen/Parents follow-ups
    senior_keywords = ["senior", "elderly", "parents", "50+", "60+", "above 50", "above 60", "old"]
    if any(k in q for k in senior_keywords):
        expansions.append("health insurance for senior citizens above 50")

    # 3. Budget/Cheapest follow-ups
    budget_keywords = ["cheapest", "budget", "affordable", "low cost", "cheap", "low premium"]
    if any(k in q for k in budget_keywords):
        expansions.append("low premium affordable health insurance plan")

    # 4. Coverage follow-ups
    coverage_keywords = ["coverage", "maximum coverage", "best coverage", "highest coverage", "comprehensive"]
    if any(k in q for k in coverage_keywords):
        expansions.append("highest sum insured comprehensive health insurance")

    # 5. Copay/Co-payment follow-ups
    copay_keywords = ["no copay", "no co-pay", "without copay", "zero copay", "no co-payment"]
    if any(k in q for k in copay_keywords):
        expansions.append("health insurance plan with no co-payment")

    if expansions:
        expanded = query + " " + " ".join(expansions)
        print(f"Query expansion applied: '{query}' -> '{expanded}'")
        return expanded
    return query


# ── Python-side Cosine Similarity Fallback (fixing RPC centoid pruning) ───────
_cached_chunks: list[dict] | None = None

def _get_all_chunks() -> list[dict]:
    """Load all policy chunks from Supabase in-memory and cache them."""
    global _cached_chunks
    if _cached_chunks is not None:
        return _cached_chunks

    print("Loading all policy chunks from Supabase into memory for python-side fallback search...")
    try:
        res = db.table("policy_chunks").select("policy_name, chunk_text, embedding").execute()
        chunks = []
        for row in (res.data or []):
            emb_val = row["embedding"]
            if not emb_val:
                continue

            if isinstance(emb_val, str):
                cleaned_val = emb_val.strip()
                if cleaned_val.startswith("[") and cleaned_val.endswith("]"):
                    emb_list = json.loads(cleaned_val)
                elif cleaned_val.startswith("{") and cleaned_val.endswith("}"):
                    emb_list = [float(x) for x in cleaned_val[1:-1].split(",")]
                else:
                    emb_list = [float(x) for x in cleaned_val.strip("[]{}").split(",")]
            else:
                emb_list = [float(x) for x in emb_val]

            chunks.append({
                "text": row["chunk_text"],
                "policy": row["policy_name"],
                "embedding": np.array(emb_list, dtype=np.float32)
            })
        _cached_chunks = chunks
        print(f"Loaded {len(_cached_chunks)} chunks into memory.")
        return _cached_chunks
    except Exception as e:
        print(f"Error loading chunks from Supabase: {e}")
        return []


# ── Search ────────────────────────────────────────────────────────────────────
def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return top-k relevant chunks from Supabase using cosine similarity."""
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    # Try RPC search first
    results = []
    try:
        result = db.rpc(
            "match_policy_chunks",
            {"query_embedding": query_embedding.tolist(), "match_count": top_k},
        ).execute()
        if result.data:
            results = [
                {
                    "text":   row["chunk_text"],
                    "policy": row["policy_name"],
                    "score":  float(row["similarity"]),
                }
                for row in result.data
                if float(row["similarity"]) >= MIN_SIM
            ]
    except Exception as e:
        print(f"Search RPC error: {e}")

    # Fallback if RPC search returned nothing or failed
    if not results:
        print("RPC search returned no results. Performing Python-side cosine similarity search...")
        chunks = _get_all_chunks()
        if chunks:
            scored_chunks = []
            for c in chunks:
                sim = float(np.dot(query_embedding, c["embedding"]))
                if sim >= MIN_SIM:
                    scored_chunks.append({
                        "text": c["text"],
                        "policy": c["policy"],
                        "score": sim
                    })
            scored_chunks.sort(key=lambda x: x["score"], reverse=True)
            results = scored_chunks[:top_k]
            if results:
                print(f"Python-side search returned {len(results)} results (best score: {results[0]['score']:.4f})")
            else:
                print("Python-side search returned 0 results matching similarity threshold.")

    return results


def search_by_policy(policy_name: str, top_k: int = 6) -> list[dict]:
    """Fetch chunks for a specific known policy by name."""
    try:
        result = (
            db.table("policy_chunks")
            .select("chunk_text, policy_name")
            .ilike("policy_name", f"%{policy_name.replace(' ', '%')}%")
            .limit(top_k)
            .execute()
        )
        return [
            {"text": row["chunk_text"], "policy": row["policy_name"], "score": 1.0}
            for row in (result.data or [])
        ]
    except Exception as e:
        print(f"Policy search error: {e}")
        return []


# ── Greeting detection and mapping ───────────────────────────────────────────
_RAG_GREETINGS = {
    # Standard Greetings
    "hi": "Hello! 👋 Welcome to Star Health Insurance. How can I help you today? Ask me about our health insurance policies, coverage options, or premium estimates. 😊",
    "hello": "Hello! 👋 Welcome to Star Health Insurance. How can I help you today? Ask me about our health insurance policies, coverage options, or premium estimates. 😊",
    "hey": "Hey there! 👋 Welcome to Star Health Insurance. Ask me anything about our plans, network hospitals, or claims processing! 😊",
    "good morning": "Good morning! ☀️ Welcome to Star Health Insurance. How can I assist you with our health insurance plans today? 😊",
    "good afternoon": "Good afternoon! ☀️ Welcome to Star Health Insurance. How can I assist you with our health insurance plans today? 😊",
    "good noon": "Good day! ☀️ Welcome to Star Health Insurance. How can I assist you with our health insurance plans today? 😊",
    "good evening": "Good evening! 🌆 Welcome to Star Health Insurance. How can I assist you with our health insurance plans today? 😊",
    "good night": "Good night! 🌙 Welcome to Star Health Insurance. If you need any assistance tomorrow, feel free to ask. Have a great sleep! 😊",

    # Acknowledgement responses
    "ok": "Great! Feel free to ask anything about Star Health Insurance. 😊",
    "okay": "Great! Feel free to ask anything about Star Health Insurance. 😊",
    "k": "Great! Feel free to ask anything about Star Health Insurance. 😊",
    "ok done": "Great! Feel free to ask anything about Star Health Insurance. 😊",
    "okkk": "Great! Feel free to ask anything about Star Health Insurance. 😊",
    "okk": "Great! Feel free to ask anything about Star Health Insurance. 😊",
    "ohk": "Great! Feel free to ask anything about Star Health Insurance. 😊",
    "ohkay": "Great! Feel free to ask anything about Star Health Insurance. 😊",

    # Thank you variations (with typos)
    "thank you": "You're welcome! 😊 Feel free to ask if you have any other questions about our plans.",
    "thankyou": "You're welcome! 😊 Feel free to ask if you have any other questions about our plans.",
    "thanks": "You're welcome! 😊 Feel free to ask if you have any other questions about our plans.",
    "thank yoh": "You're welcome! 😊 Feel free to ask if you have any other questions about our plans.",
    "thank uh": "You're welcome! 😊 Feel free to ask if you have any other questions about our plans.",
    "thnks": "You're welcome! 😊 Feel free to ask if you have any other questions about our plans.",
    "thnx": "You're welcome! 😊 Feel free to ask if you have any other questions about our plans.",
    "thx": "You're welcome! 😊 Feel free to ask if you have any other questions about our plans.",
    "ty": "You're welcome! 😊 Feel free to ask if you have any other questions about our plans.",
    "thanku": "You're welcome! 😊 Feel free to ask if you have any other questions about our plans.",
    "thankyu": "You're welcome! 😊 Feel free to ask if you have any other questions about our plans.",
    "tank you": "You're welcome! 😊 Feel free to ask if you have any other questions about our plans.",

    # OK + Thank you combinations
    "ok thank you": "You're welcome! 😊 We're always here to help. Have a great day! 🌟",
    "ok thanks": "You're welcome! 😊 We're always here to help. Have a great day! 🌟",
    "okay thanks": "You're welcome! 😊 We're always here to help. Have a great day! 🌟",
    "ok thankyou": "You're welcome! 😊 We're always here to help. Have a great day! 🌟",
    "ok thank yoh": "You're welcome! 😊 We're always here to help. Have a great day! 🌟",

    # Bye variations
    "bye": "Goodbye! 👋 Thank you for contacting Star Health Insurance. Have a healthy day ahead! 😊",
    "byee": "Goodbye! 👋 Thank you for contacting Star Health Insurance. Have a healthy day ahead! 😊",
    "byebye": "Goodbye! 👋 Thank you for contacting Star Health Insurance. Have a healthy day ahead! 😊",
    "ok bye": "Goodbye! 👋 Thank you for contacting Star Health Insurance. Have a healthy day ahead! 😊",
    "okay bye": "Goodbye! 👋 Thank you for contacting Star Health Insurance. Have a healthy day ahead! 😊",
    "goodbye": "Goodbye! 👋 Thank you for contacting Star Health Insurance. Have a healthy day ahead! 😊",
    "good bye": "Goodbye! 👋 Thank you for contacting Star Health Insurance. Have a healthy day ahead! 😊",
    "see you": "Goodbye! 👋 Thank you for contacting Star Health Insurance. Have a healthy day ahead! 😊",
    "ttyl": "Goodbye! 👋 Thank you for contacting Star Health Insurance. Have a healthy day ahead! 😊",
    "cya": "Goodbye! 👋 Thank you for contacting Star Health Insurance. Have a healthy day ahead! 😊",

    # Positive acknowledgements
    "got it": "Great! Let me know if you have more questions. 😊",
    "noted": "Great! Let me know if you have more questions. 😊",
    "understood": "Great! Let me know if you have more questions. 😊",
    "alright": "Great! Let me know if you have more questions. 😊",
    "sure": "Great! Let me know if you have more questions. 😊",
    "sounds good": "Great! Let me know if you have more questions. 😊",

    # Filler/thinking messages
    "aah": "Take your time! Feel free to ask anything about our Star Health Insurance plans. 😊",
    "ahh": "Take your time! Feel free to ask anything about our Star Health Insurance plans. 😊",
    "hmm": "Take your time! Feel free to ask anything about our Star Health Insurance plans. 😊",
    "hmmm": "Take your time! Feel free to ask anything about our Star Health Insurance plans. 😊",
    "oh": "Take your time! Feel free to ask anything about our Star Health Insurance plans. 😊",
    "ohh": "Take your time! Feel free to ask anything about our Star Health Insurance plans. 😊",
    "oh okay": "Take your time! Feel free to ask anything about our Star Health Insurance plans. 😊",
    "oh ok": "Take your time! Feel free to ask anything about our Star Health Insurance plans. 😊",
    "i see": "Take your time! Feel free to ask anything about our Star Health Insurance plans. 😊",

    # Positive reactions
    "nice": "Glad to hear that! 😊 Feel free to ask anything about Star Health Insurance — policies, benefits, network hospitals, or claims.",
    "great": "Glad to hear that! 😊 Feel free to ask anything about Star Health Insurance — policies, benefits, network hospitals, or claims.",
    "good": "Glad to hear that! 😊 Feel free to ask anything about Star Health Insurance — policies, benefits, network hospitals, or claims.",
    "wow": "Glad to hear that! 😊 Feel free to ask anything about Star Health Insurance — policies, benefits, network hospitals, or claims.",
    "cool": "Glad to hear that! 😊 Feel free to ask anything about Star Health Insurance — policies, benefits, network hospitals, or claims.",
    "awesome": "Glad to hear that! 😊 Feel free to ask anything about Star Health Insurance — policies, benefits, network hospitals, or claims.",
    "wonderful": "Glad to hear that! 😊 Feel free to ask anything about Star Health Insurance — policies, benefits, network hospitals, or claims.",
    "perfect": "Glad to hear that! 😊 Feel free to ask anything about Star Health Insurance — policies, benefits, network hospitals, or claims.",
    "excellent": "Glad to hear that! 😊 Feel free to ask anything about Star Health Insurance — policies, benefits, network hospitals, or claims.",

    # Yes/No simple responses
    "yes": "Sure! Please go ahead and ask your question. 😊",
    "yeah": "Sure! Please go ahead and ask your question. 😊",
    "yep": "Sure! Please go ahead and ask your question. 😊",
    "yup": "Sure! Please go ahead and ask your question. 😊",
    "ya": "Sure! Please go ahead and ask your question. 😊",
    "yaa": "Sure! Please go ahead and ask your question. 😊",
    "haan": "Sure! Please go ahead and ask your question. 😊",
    "no": "No problem! Feel free to reach out anytime if you need help with our insurance plans. 😊",
    "nope": "No problem! Feel free to reach out anytime if you need help with our insurance plans. 😊",
    "nah": "No problem! Feel free to reach out anytime if you need help with our insurance plans. 😊",
    "nahi": "No problem! Feel free to reach out anytime if you need help with our insurance plans. 😊",
}


def normalize_greeting(text: str) -> Optional[str]:
    """Clean and normalize greeting input to match standard greeting responses."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower().strip())
    cleaned = re.sub(r"\s+", " ", cleaned)

    if cleaned in _RAG_GREETINGS:
        return cleaned

    # Check regex patterns for standard greetings
    if re.match(r"^hi+$", cleaned):
        return "hi"
    if re.match(r"^he+y+$", cleaned):
        return "hey"
    if re.match(r"^hello+$|^helo+$", cleaned):
        return "hello"

    # OK + Thank you combinations
    if re.match(r"^(ok|okay)\s+(thank\s*(you|yoh|uh|yu|u)|thanks|thnks|thnx|thx|ty|thanku)$", cleaned):
        return "ok thank you"

    # Thank you variations
    if re.match(r"^(thank\s*(you|yoh|uh|yu|u)|thanks|thnks|thnx|thx|ty|thanku|thankyu|tank\s*you)$", cleaned):
        return "thank you"

    # Bye variations
    if re.match(r"^(ok\s+|okay\s+)?(bye|byee|byebye|bbye|bye\s+bye|goodbye|good\s*bye|see\s*you|ttyl|cya)$", cleaned):
        return "bye"

    # OK / Acknowledgement variations
    if re.match(r"^(ok+|okay|k|ohk|ohkay|ok\s+done|okay\s+done)$", cleaned):
        return "ok"

    # Positive acknowledgement variations
    if re.match(r"^(got\s*it|noted|understood|alright|sure|sounds\s*good)$", cleaned):
        return "got it"

    # Filler variations
    if re.match(r"^(aa+h|ah+h|hm+m|hmm+m|oh+|oh+h|oh\s+okay|oh\s+ok|i\s+see)$", cleaned):
        return "aah"

    # Positive reaction variations
    if re.match(r"^(nice|great|good|wow|cool|awesome|wonderful|perfect|excellent)$", cleaned):
        return "nice"

    # Yes variations
    if re.match(r"^(yes|yeah|yep|yup|ya+|haan)$", cleaned):
        return "yes"

    # No variations
    if re.match(r"^(no|nope|nah|nahi)$", cleaned):
        return "no"

    return None


def _check_and_build_greeting(text: str) -> Optional[str]:
    """Check if the text is a pure greeting/chitchat and construct response. Excludes queries with insurance or question keywords."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower().strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    words = set(cleaned.split())
    if not words:
        return None

    # Strict list of insurance keywords
    insurance_keywords = {
        "policy", "policies", "plan", "plans", "insurance", "cover", "coverage", 
        "hospital", "hospitals", "waiting", "period", "disease", "diseases", 
        "premium", "premiums", "cost", "price", "prices", "claim", "claims", 
        "arogya", "sanjeevani", "optima", "premier", "assure", "young", "classic", 
        "maternity", "cashless", "copay", "co-pay", "rent", "room", "limit", "ped", "pre-existing"
    }
    # List of question words
    question_words = {"what", "how", "which", "when", "where", "why", "who", "whose", "whom"}

    if words.intersection(insurance_keywords) or words.intersection(question_words):
        return None

    norm = normalize_greeting(text)
    if norm:
        return _RAG_GREETINGS[norm]

    return None


# ── Ask ───────────────────────────────────────────────────────────────────────
def ask(question: str, channel: str = "web") -> str:
    """Retrieve relevant context and generate an answer using Groq LLM."""
    question = question.strip()
    if not question:
        return "Please ask a question about Star Health Insurance."

    # ── Check for greeting ────────────────────────────────────────────────────
    greeting_reply = _check_and_build_greeting(question)
    if greeting_reply:
        return greeting_reply

    # ── Special case: broad "list all policies" query ─────────────────────────
    if _is_list_query(question):
        # Combine the static summary with any retrieved context
        relevant_chunks = search(question, top_k=4)
        context_extra = ""
        if relevant_chunks:
            context_extra = "\n\nAdditional context from policy documents:\n"
            for c in relevant_chunks[:3]:
                context_extra += f"\n[{c['policy']}]: {c['text'][:300]}...\n"

        prompt = f"""You are a knowledgeable Star Health Insurance advisor.
The user wants to know about all available insurance policies.

Here is the complete list of Star Health Insurance policies:
{POLICY_SUMMARIES}
{context_extra}

Question: {question}

Provide a comprehensive, well-formatted answer listing ALL the policies with their key features.
Use bullet points or numbered lists for clarity. Be friendly and helpful. 
Do NOT include any PDF links or download URLs in your response."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500,
        )
        reply_text = response.choices[0].message.content

        # Programmatically attach download links for WhatsApp only
        if channel == "whatsapp":
            links = []
            for key, url in POLICY_PDF_MAP.items():
                links.append(f"📥 Download {key.title()} PDF: {url}")
            reply_text += "\n\n📄 Policy Downloads:\n" + "\n".join(links)

        return reply_text

    # ── Apply Query Expansion for Follow-up Questions ───────────────────────
    search_query = _expand_query(question)

    # ── Try vector search first ───────────────────────────────────────────────
    relevant_chunks = search(search_query, top_k=TOP_K)

    # ── If vector search gives poor results, try keyword-based policy search ──
    if len(relevant_chunks) < 2:
        policy_mentioned = _extract_policy_keyword(question)
        if policy_mentioned:
            keyword_chunks = search_by_policy(policy_mentioned, top_k=6)
            # Merge, deduplicate by text
            existing_texts = {c["text"] for c in relevant_chunks}
            for c in keyword_chunks:
                if c["text"] not in existing_texts:
                    relevant_chunks.append(c)
                    existing_texts.add(c["text"])

    # ── Still no results → general fallback ──────────────────────────────────
    if not relevant_chunks:
        # Try a broader search with just key terms
        key_terms = " ".join(search_query.split()[:5])
        relevant_chunks = search(key_terms, top_k=4)

    if not relevant_chunks:
        return (
            "I couldn't find specific information about that in our policy documents. "
            "However, Star Health Insurance offers 7 main plans:\n\n"
            f"{POLICY_SUMMARIES}\n\n"
            "Please ask about any specific plan or call us at 1800-425-2255 for personalized assistance."
        )

    # ── Build context with clean policy labels ────────────────────────────────
    # Group by policy for structured context
    policy_context: dict[str, list[str]] = {}
    for chunk in relevant_chunks:
        pname = chunk["policy"]
        # Clean up long file-hash names
        pname = re.sub(r"_[a-f0-9]{8,}.*$", "", pname, flags=re.IGNORECASE)
        pname = re.sub(r"\s*\(1\)\s*", "", pname)
        if pname not in policy_context:
            policy_context[pname] = []
        # Strip the "[Policy: name]" header if present in chunk text
        text = re.sub(r"^\[Policy:[^\]]+\]\s*", "", chunk["text"])
        policy_context[pname].append(text[:600])

    context_parts = []
    for pname, texts in policy_context.items():
        context_parts.append(f"=== {pname} ===")
        for t in texts[:3]:
            context_parts.append(t)

    context = "\n\n".join(context_parts)

    # ── Generate answer ───────────────────────────────────────────────────────
    prompt = f"""You are a helpful and knowledgeable Star Health Insurance advisor.
Below is a brief summary of all available Star Health Insurance policies, followed by specific document excerpts.
Use this information to answer the customer's question comprehensively.

STAR HEALTH POLICY SUMMARIES:
{POLICY_SUMMARIES}

POLICY DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
- Answer based on the provided context and policy summaries. If partial information is available, use it and say what you know.
- Do NOT say "I don't have that information" if any relevant context or summary exists — synthesize what's available.
- If the question is about a specific policy, focus on that policy's details.
- If the user asks for a recommendation or comparison (e.g. "which is best", "which one is good for family", "best plan for someone above 60", "which plan has no copay"), you MUST analyze the retrieved context and summaries, and declare a CLEAR WINNER (or winners) with a direct recommendation. Do NOT just list all policies or give a generic fallback. Name the specific policy and explain exactly why it fits the criteria (e.g., Family Health Optima for families, Star Health Premier for 50+, Medi Classic or Super Star for no co-pay) based on the features in the context and summaries.
- Do NOT include any PDF download links or document URLs in your answer. These are not needed in this chat interface.
- Format your answer clearly with bullet points or sections where appropriate.
- Always be helpful, professional, and specific.
- Mention the policy name(s) your answer is based on.

CUSTOMER QUESTION: {question}

ANSWER:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
    )
    reply_text = response.choices[0].message.content

    # ── Programmatic PDF Link Attachment for WhatsApp ────────────────────────
    if channel == "whatsapp":
        policy_mentioned = _extract_policy_keyword(question)
        if policy_mentioned:
            pdf_url = POLICY_PDF_MAP.get(policy_mentioned.lower())
            if pdf_url:
                reply_text += f"\n\n📄 Download policy PDF: {pdf_url}"
        else:
            # Check if any policy name keyword is present in the LLM response text
            for keyword, name in POLICY_KEYWORDS.items():
                if keyword in reply_text.lower():
                    pdf_url = POLICY_PDF_MAP.get(keyword)
                    if pdf_url:
                        reply_text += f"\n\n📄 Download {name} PDF: {pdf_url}"
                        break

    return reply_text


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "hii",
        "good morning",
        "Give me all types of insurance policies",
        "Tell me about Arogya Sanjeevani",
        "Which policy is best for a family?",
        "What is the waiting period for pre-existing diseases?",
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        answer = ask(q)
        print(answer)
        print()