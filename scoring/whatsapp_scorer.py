"""
scoring/whatsapp_scorer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WhatsApp Conversation Behavior Scorer using Groq LLM (llama-3.1-8b-instant).
Evaluates incoming message for buying intent, detailed questions, or negative signals,
updates the cumulative delta, runs the blending engine, and updates Supabase.
"""

from __future__ import annotations
import os
import re
import json
import logging
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
from groq import Groq
from conversation_store import _get_supabase, normalise_phone, phone_variants
from scoring.engine import calculate_blended_score, classify_lead_type

logger = logging.getLogger("scoring.whatsapp_scorer")

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _get_groq_client() -> Groq:
    load_dotenv(override=True)
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    return Groq(api_key=api_key)


def _analyze_whatsapp_message(message: str) -> dict:
    """
    Call Groq to analyze the WhatsApp message and classify intent/delta.
    """
    client = _get_groq_client()
    prompt = f"""You are a lead intent analyzer for Star Health Insurance.
Analyze this WhatsApp message from a potential customer and return JSON:
{{
  "intent": "buying|inquiring|negative|neutral",
  "score_delta": <integer between -20 and +15>,
  "reason": "<one line explanation>"
}}
Message: "{message}"
Return ONLY valid JSON, no markdown.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
        )
        content = response.choices[0].message.content.strip()
        # Clean any markdown code wrappers
        if content.startswith("```"):
            content = re.sub(r"^```(json)?\n", "", content, flags=re.IGNORECASE)
            content = re.sub(r"\n```$", "", content)
        
        return json.loads(content.strip())
    except Exception as exc:
        logger.error(f"Groq analysis failed for message: {exc}")
        return {
            "intent": "neutral",
            "score_delta": 0,
            "reason": f"Fallback: analysis error ({exc})"
        }


def _update_lead_score_sync(phone: str, message_body: str) -> None:
    """
    Synchronous DB operations for updating lead score based on WhatsApp message.
    Called inside a thread pool to avoid blocking the event loop.
    """
    try:
        db = _get_supabase()
        variants = phone_variants(phone)
        
        # 1. Resolve lead
        res_lead = db.table("leads").select("*").in_("phone", variants).order("created_at", desc=True).limit(1).execute()
        rows = res_lead.data or []
        if not rows:
            logger.warning(f"No lead resolved for phone variants {variants}. Skipping score update.")
            return
        
        lead = rows[0]
        lead_id = lead["id"]
        
        # 2. Check for daily follow-up bonus
        # Count inbound messages for this lead on the current calendar day (UTC)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        res_msgs = db.table("messages").select("id").eq("lead_id", lead_id).eq("direction", "inbound").gte("created_at", f"{today_str}T00:00:00Z").execute()
        inbound_count_today = len(res_msgs.data or [])
        
        # Since record_message is called before scoring, the current message is already in the DB.
        # If count > 1, then this is a follow-up message today.
        is_followup = inbound_count_today > 1
        
        # 3. Call Groq for intent analysis
        analysis = _analyze_whatsapp_message(message_body)
        intent = analysis.get("intent", "neutral")
        score_delta = int(analysis.get("score_delta", 0))
        reason = analysis.get("reason", "Analyzed WhatsApp message")
        
        # Apply follow-up bonus
        msg_delta = score_delta
        if is_followup:
            msg_delta += 5
            logger.info(f"Applying +5 daily follow-up bonus for lead_id={lead_id}")
            
        current_cumulative_delta = lead.get("whatsapp_score_delta") or 0
        new_cumulative_delta = current_cumulative_delta + msg_delta
        
        # Clamp cumulative delta so the WhatsApp behavior component is bounded [0, 100]
        whatsapp_behavior_score = max(0.0, min(100.0, 70.0 + new_cumulative_delta))
        # Recalculate normalized cumulative delta based on clamped score
        new_cumulative_delta = int(whatsapp_behavior_score - 70.0)
        
        # 4. Blend scores
        profile_score = lead.get("profile_score")
        if profile_score is None:
            # Fallback to current ai_rank_score if profile_score was not backfilled
            profile_score = lead.get("ai_rank_score")
            
        call_score = lead.get("call_score")
        
        final_score = calculate_blended_score(
            profile_score = profile_score,
            whatsapp_behavior = whatsapp_behavior_score,
            call_signal = call_score
        )
        
        new_lead_type = classify_lead_type(final_score)
        
        # 5. Update leads table
        db.table("leads").update({
            "whatsapp_score_delta": new_cumulative_delta,
            "last_whatsapp_intent": intent,
            "ai_rank_score": int(final_score),
            "lead_type": new_lead_type,
            "score_updated_at": _now_iso(),
            "ai_rank_explanation": f"WhatsApp intent: {intent}. Delta applied: {msg_delta} (reason: {reason}). Blended score updated."
        }).eq("id", lead_id).execute()
        
        logger.info(
            f"Successfully updated lead_id={lead_id}: old_score={lead.get('ai_rank_score')} -> "
            f"new_score={int(final_score)} ({new_lead_type}), cumulative_delta={new_cumulative_delta}"
        )
        
    except Exception as e:
        logger.error(f"Failed to update lead score from WhatsApp for phone {phone}: {e}", exc_info=True)


async def update_lead_score_from_whatsapp(phone: str, message_body: str) -> None:
    """
    Asynchronously update the lead score from WhatsApp message.
    """
    logger.info(f"Triggering WhatsApp score update for phone {phone}...")
    await asyncio.to_thread(_update_lead_score_sync, phone, message_body)
