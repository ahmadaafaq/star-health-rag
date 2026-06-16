import os
import sys
import asyncio
import urllib.parse
import httpx
import logging
import uuid
import re
from typing import List, Optional
from fastapi import FastAPI, Form, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

# Twilio validation and client
from twilio.request_validator import RequestValidator

# Import the RAG ask function and greeting helpers
from rag import ask, normalize_greeting, _RAG_GREETINGS

# Import database storage and normalisation functions
from conversation_store import (
    record_message,
    get_conversation_history,
    normalise_phone,
    phone_variants,
    _get_supabase,
    logger
)

# Import lead scoring modules
from scoring.whatsapp_scorer import update_lead_score_from_whatsapp
from scoring.call_scorer import update_lead_score_from_call

load_dotenv(override=True)
# Load from web env as well if present
web_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "star-health-web", ".env")
if os.path.exists(web_env_path):
    load_dotenv(dotenv_path=web_env_path, override=False)

app = FastAPI(
    title="Star Health Insurance Assistant & WhatsApp Webhook",
    version="1.0.0"
)

# Allow the Node.js frontend server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
#  POLICY PDF MAP — Supabase public storage URLs
# ═══════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Dedup set for Twilio message SIDs to prevent processing the same webhook twice
_processed_sids = set()

def _twiml_empty() -> str:
    """Return an empty TwiML response."""
    return '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'

def is_pure_greeting(body: str) -> bool:
    """
    Check if the user message is a pure greeting or simple chitchat.
    Returns True if yes, False if it is a substantive query.
    It should match exact single-word greetings and spelling variations (hii, hello, thanks, bye, good morning, etc.).
    Any message containing a question or insurance keyword must go to RAG.
    """
    cleaned = re.sub(r"[^\w\s]", "", body.lower().strip())
    words = set(cleaned.split())
    if not words:
        return False

    insurance_keywords = {
        "policy", "policies", "plan", "plans", "insurance", "cover", "coverage", 
        "hospital", "hospitals", "waiting", "period", "disease", "diseases", 
        "premium", "premiums", "cost", "price", "prices", "claim", "claims", 
        "arogya", "sanjeevani", "optima", "premier", "assure", "young", "classic", 
        "maternity", "cashless", "copay", "co-pay", "rent", "room", "limit", "ped", "pre-existing"
    }
    question_words = {"what", "how", "when", "where", "which", "who", "why", "whose", "whom"}

    if words.intersection(insurance_keywords) or words.intersection(question_words):
        return False

    norm = normalize_greeting(body)
    return norm is not None

def _build_auto_reply(body: str) -> str:
    """Generate automatic greeting or chitchat fallback replies."""
    norm = normalize_greeting(body)
    if norm and norm in _RAG_GREETINGS:
        return _RAG_GREETINGS[norm]
    return "Hello! 👋 Welcome to Star Health Insurance. How can I assist you with our health insurance plans today? 😊"

async def validate_twilio_signature(request: Request) -> bool:
    """Validate that the incoming request is indeed from Twilio."""
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN") or ""
    if not auth_token:
        logger.warning("TWILIO_AUTH_TOKEN not configured. Skipping Twilio signature validation.")
        return True

    signature = request.headers.get("X-Twilio-Signature", "")
    if not signature:
        logger.warning("Missing X-Twilio-Signature header.")
        return False

    url = str(request.url)
    # Handle proxy headers (ngrok)
    forwarded_proto = request.headers.get("X-Forwarded-Proto")
    forwarded_host = request.headers.get("X-Forwarded-Host")
    if forwarded_proto and forwarded_host:
        url = f"{forwarded_proto}://{forwarded_host}{request.url.path}"
    elif forwarded_proto:
        if url.startswith("http://"):
            url = url.replace("http://", f"{forwarded_proto}://")
        elif url.startswith("https://"):
            url = url.replace("https://", f"{forwarded_proto}://")

    # Retrieve form data
    form_data = await request.form()
    data = {k: v for k, v in form_data.items()}

    validator = RequestValidator(auth_token)
    is_valid = validator.validate(url, data, signature)
    if not is_valid:
        logger.warning(f"Twilio signature validation failed. URL={url}, Params={data}")
    return is_valid

async def _send_twilio_reply(to: str, body: str, media_url: Optional[str] = None, depth: int = 0) -> Optional[str]:
    """Send an outbound WhatsApp message using Twilio's Messages API."""
    # Split/Truncate the message if it exceeds Twilio 1600 character limit
    if len(body) > 1600:
        if depth >= 1:
            logger.info(f"Message exceeds 1600 characters at depth {depth}. Truncating to 1600 chars.")
            body = body[:1585] + " (truncated)"
        else:
            logger.info(f"Message exceeds 1600 characters (len={len(body)}). Splitting and sending in parts.")
            part1 = body[:1500]
            part2 = body[1500:]
            sid1 = await _send_twilio_reply(to, part1, media_url, depth=depth + 1)
            await asyncio.sleep(0.5)
            sid2 = await _send_twilio_reply(to, part2, None, depth=depth + 1)
            return sid1 or sid2

    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_whatsapp = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    test_number = os.environ.get("TEST_PERSONAL_MOBILE_NUMBER")

    # Handle test number overrides if configured
    target_to = to
    if test_number:
        norm_to = normalise_phone(to)
        norm_test = normalise_phone(test_number)
        if norm_to != norm_test:
            logger.info(f"Test mode redirect: original recipient={to}, redirecting to {test_number}")
            target_to = test_number

    # Ensure phone has "whatsapp:" prefix
    if not target_to.startswith("whatsapp:"):
        target_to = f"whatsapp:{target_to}"

    if not account_sid or not auth_token:
        # Generate mock SID and log instead of failing silently when credentials are missing
        mock_sid = f"SMmock_{uuid.uuid4().hex[:16]}"
        logger.warning(f"[MOCK TWILIO] Credentials missing. Logging mock message delivery. to={target_to}, body={body[:100]}..., media_url={media_url}, mock_sid={mock_sid}")
        return mock_sid

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    payload = {
        "From": from_whatsapp,
        "To": target_to,
        "Body": body,
    }
    if media_url:
        payload["MediaUrl"] = media_url

    auth = (account_sid, auth_token)

    logger.info(f"Triggering outbound Twilio call to {target_to}: from={from_whatsapp}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, data=payload, auth=auth)
            if response.status_code in (200, 201):
                res_data = response.json()
                sid = res_data.get("sid")
                logger.info(f"Twilio outbound message sent successfully! SID={sid}")
                return sid
            else:
                logger.error(f"Twilio API error (Status {response.status_code}): {response.text}")
                return None
        except Exception as e:
            logger.error(f"Exception encountered while calling Twilio API: {e}", exc_info=True)
            return None

def resolve_policy_pdf(policy_name: Optional[str], plan_id: Optional[str]) -> tuple[str, str]:
    """
    Resolve policy name or plan ID to the standard PDF URL and policy title.
    Returns (pdf_url, matched_policy_name)
    """
    # Normalize inputs
    p_name = (policy_name or "").strip().lower().replace("-", " ")
    p_id = (plan_id or "").strip().lower().replace("-", " ")
    combined = f"{p_name} {p_id}"

    # Map keywords to key in POLICY_PDF_MAP and clean title
    mapping = [
        ("arogya", "arogya sanjeevani", "Arogya Sanjeevani"),
        ("sanjeevani", "arogya sanjeevani", "Arogya Sanjeevani"),
        ("optima", "family health optima", "Family Health Optima"),
        ("fho", "family health optima", "Family Health Optima"),
        ("medi classic", "medi classic", "Medi Classic (Individual)"),
        ("classic", "medi classic", "Medi Classic (Individual)"),
        ("assure", "star assure", "Star Health Assure"),
        ("premier", "star premier", "Star Health Premier"),
        ("young star", "young star", "Young Star Insurance"),
        ("super star", "super star", "Super Star"),
    ]

    for keyword, map_key, title in mapping:
        if keyword in combined:
            return POLICY_PDF_MAP[map_key], title

    # Fallback to Young Star
    return POLICY_PDF_MAP["young star"], "Young Star Insurance"

# ═══════════════════════════════════════════════════════════════════════════════
#  REQUEST SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class ManualReplyRequest(BaseModel):
    phone: str
    message: str
    mediaUrl: Optional[str] = None

class WelcomeRequest(BaseModel):
    phone: str
    name: str
    policy_name: Optional[str] = None
    recommended_plan_id: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    phone: Optional[str] = None
    name: Optional[str] = None

class CallScoreRequest(BaseModel):
    lead_id: str
    call_summary: str

# ═══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/webhook/whatsapp")
async def webhook_whatsapp(
    request: Request,
    MessageSid: str = Form(None),
    Body: str = Form(None),
    From: str = Form(None),
    To: str = Form(None),
    ProfileName: str = Form("WhatsApp User")
):
    """
    Twilio WhatsApp incoming webhook.
    Processes user messages, queries the RAG system, sends a reply via Twilio,
    and triggers fire-and-forget background lead scoring.
    """
    logger.info(f"Inbound WhatsApp webhook received: MessageSid={MessageSid}, From={From}, Body={Body}")

    # 1. Deduplication
    if MessageSid:
        if MessageSid in _processed_sids:
            logger.info(f"Duplicate MessageSid={MessageSid} detected. Skipping webhook execution.")
            return Response(content=_twiml_empty(), media_type="application/xml")
        _processed_sids.add(MessageSid)
        if len(_processed_sids) > 1000:
            _processed_sids.pop()

    # 2. Signature verification
    validate_sig = os.environ.get("TWILIO_VALIDATE_SIGNATURE", "false").lower() == "true"
    if validate_sig:
        is_valid = await validate_twilio_signature(request)
        if not is_valid:
            logger.warning(f"Signature validation failed for webhook from {From}")
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    norm_phone = normalise_phone(From)

    # 3. Record inbound message
    await record_message(
        lead_phone=From,
        lead_name=ProfileName or "WhatsApp User",
        direction="inbound",
        body=Body or "",
        channel="whatsapp",
        sid=MessageSid
    )

    # 4. Trigger WhatsApp Scorer as fire-and-forget task
    if Body:
        logger.info(f"Spawning lead scorer background task for phone {norm_phone}")
        asyncio.create_task(update_lead_score_from_whatsapp(norm_phone, Body))

    # 5. Routing decision
    is_greeting = is_pure_greeting(Body or "")
    logger.info(f"Routing decision: is_pure_greeting={is_greeting} for text: '{Body}'")

    if is_greeting:
        reply_text = _build_auto_reply(Body or "")
    else:
        # Detailed logs before and after ask()
        logger.info(f"[Webhook Routing] Querying RAG ask() for body: '{Body}'")
        reply_text = await asyncio.to_thread(ask, Body or "", "whatsapp")
        logger.info(f"[Webhook Routing] RAG query completed. Reply length: {len(reply_text)}")

    # 6. Send Twilio reply
    logger.info(f"[Webhook Routing] Triggering _send_twilio_reply to={From}, body length={len(reply_text)}")
    outbound_sid = await _send_twilio_reply(to=From, body=reply_text)
    if outbound_sid:
        logger.info(f"[Webhook Routing] Twilio reply sent successfully. outbound_sid={outbound_sid}")
    else:
        logger.error(f"[Webhook Routing] Twilio reply failed to send for phone={From}")

    # 7. Record outbound message
    if outbound_sid:
        await record_message(
            lead_phone=From,
            lead_name=ProfileName or "WhatsApp User",
            direction="outbound",
            body=reply_text,
            channel="whatsapp",
            sid=outbound_sid
        )

    # Return empty TwiML response to satisfy Twilio webhook requirements
    return Response(content=_twiml_empty(), media_type="application/xml")

@app.post("/send-reply")
async def send_reply(req: ManualReplyRequest):
    """
    Endpoint to manually send an outbound WhatsApp reply to a lead from the dashboard.
    """
    phone = req.phone
    message = req.message
    media_url = req.mediaUrl

    if not phone or not message:
        raise HTTPException(status_code=400, detail="phone and message are required parameters")

    sid = await _send_twilio_reply(to=phone, body=message, media_url=media_url)
    if not sid:
        raise HTTPException(status_code=500, detail="Failed to send outbound message via Twilio")

    # Record message
    await record_message(
        lead_phone=phone,
        lead_name="Agent / System",
        direction="outbound",
        body=message,
        channel="whatsapp",
        sid=sid
    )

    return {"status": "success", "message_sid": sid}

@app.post("/send-welcome")
async def send_welcome(req: WelcomeRequest):
    """
    Called upon submission of a new lead to send a welcome message and relevant policy document.
    """
    phone = req.phone
    name = req.name

    if not phone or not name:
        raise HTTPException(status_code=400, detail="phone and name are required parameters")

    # Resolve policy PDF mapping using the robust resolver helper
    pdf_url, matched_policy_name = resolve_policy_pdf(req.policy_name, req.recommended_plan_id)

    norm_phone = normalise_phone(phone)

    # Simple guard: check if a welcome message was already sent to this phone in the last 10 minutes
    try:
        db = _get_supabase()
        ten_mins_ago = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(timespec="seconds")
        
        # Query messages table
        res = db.table("messages").select("id").eq("phone", norm_phone).eq("message_type", "welcome").gte("created_at", ten_mins_ago).limit(1).execute()
        if res.data:
            logger.info(f"Welcome message already sent to {norm_phone} in the last 10 minutes. Skipping welcome flow.")
            return {
                "status": "success",
                "message": "Welcome message already sent recently. Skipped duplicate.",
                "welcome_sid": "skipped_duplicate",
                "pdf_sid": "skipped_duplicate",
                "policy_sent": matched_policy_name,
                "pdf_url": pdf_url
            }
    except Exception as e:
        logger.warning(f"Error checking for existing welcome message for {norm_phone}: {e}")

    welcome_msg = (
        f"Welcome to Star Health Insurance, {name}! 🌟\n\n"
        f"Thank you for exploring our plans. We have successfully registered your custom health insurance quote.\n\n"
        f"Our digital advisor is here to help you protect what matters most. We offer a wide range of plans "
        f"tailored to your family's needs, including cashless hospitalization, maternity cover, senior citizen "
        f"health plans, and pre-existing disease coverage with seamless claim settlement.\n\n"
        f"A dedicated Star Health relationship manager will contact you shortly to guide you through the next steps "
        f"and answer any questions.\n\n"
        f"In the meantime, feel free to ask me anything about our policies directly here! 😊\n\n"
        f"— Star Health Advisory Team"
    )

    pdf_msg = (
        f"📄 Here is your {matched_policy_name} policy document.\n"
        f"Feel free to review it at your convenience. Our advisor will walk you through the highlights shortly! 😊"
    )

    # 1. Send welcome message
    welcome_sid = await _send_twilio_reply(to=phone, body=welcome_msg)
    if welcome_sid:
        await record_message(
            lead_phone=phone,
            lead_name=name,
            direction="outbound",
            body=welcome_msg,
            channel="whatsapp",
            sid=welcome_sid,
            message_type="welcome"
        )
    else:
        logger.error(f"Failed to send welcome message via Twilio for lead: {name} ({phone})")

    # Add a 1 second delay between the welcome message and the PDF message to prevent order mismatch
    await asyncio.sleep(1)

    # 2. Send PDF message with attachment
    pdf_sid = await _send_twilio_reply(to=phone, body=pdf_msg, media_url=pdf_url)
    if pdf_sid:
        await record_message(
            lead_phone=phone,
            lead_name=name,
            direction="outbound",
            body=pdf_msg,
            channel="whatsapp",
            sid=pdf_sid,
            message_type="welcome"
        )
    else:
        logger.error(f"Failed to send policy PDF via Twilio for lead: {name} ({phone})")

    return {
        "status": "success" if (welcome_sid or pdf_sid) else "failed",
        "welcome_sid": welcome_sid,
        "pdf_sid": pdf_sid,
        "policy_sent": matched_policy_name,
        "pdf_url": pdf_url,
        "error_details": None if (welcome_sid and pdf_sid) else "Twilio message delivery failed (possibly daily limit exceeded or credentials issue)."
    }

@app.get("/conversation/{phone}")
async def get_conversation(phone: str, limit: int = Query(20, ge=1)):
    """
    Get full conversation history (both Database and In-memory) for a phone number.
    """
    try:
        history = await get_conversation_history(phone, limit=limit)
        return {"status": "success", "history": history}
    except Exception as e:
        logger.error(f"Error fetching conversation history for {phone}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Chat endpoint proxy used by the web client UI.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages list is empty")

    # Get the latest user message from history
    message = req.messages[-1].content
    phone = req.phone
    name = req.name or "Web User"

    # 1. Record inbound message if phone number exists
    if phone:
        await record_message(
            lead_phone=phone,
            lead_name=name,
            direction="inbound",
            body=message,
            channel="web"
        )

    # 2. Query RAG System
    logger.info(f"Querying RAG from web: '{message}'")
    reply = await asyncio.to_thread(ask, message, "web")
    logger.info(f"RAG Web response: '{reply[:100]}...'")

    # 3. Record outbound response if phone number exists
    if phone:
        await record_message(
            lead_phone=phone,
            lead_name=name,
            direction="outbound",
            body=reply,
            channel="web"
        )

    # Return structure expected by frontend server: {"message": reply}
    return {"message": reply}

@app.post("/api/update-score-from-call")
async def update_score_from_call_endpoint(req: CallScoreRequest):
    """
    Receives call summaries from agent calls, runs sentiment/intent analysis,
    calculates blended score, and updates Supabase.
    """
    lead_id = req.lead_id
    call_summary = req.call_summary

    if not lead_id or not call_summary:
        raise HTTPException(status_code=400, detail="lead_id and call_summary are required fields")

    updated_lead = await update_lead_score_from_call(lead_id, call_summary)
    if not updated_lead:
        raise HTTPException(status_code=500, detail="Failed to calculate or update score from call summary")

    return {"status": "success", "lead": updated_lead}

@app.get("/health")
async def health_check():
    """
    Health check endpoint probing database connectivity.
    """
    db_ok = False
    try:
        db = _get_supabase()
        db.table("leads").select("id").limit(1).execute()
        db_ok = True
    except Exception as e:
        logger.error(f"Health check failed to probe Supabase leads: {e}")

    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "error",
        "service": "api.py"
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  START SERVER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("RAG_PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)