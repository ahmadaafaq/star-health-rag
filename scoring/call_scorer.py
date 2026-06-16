"""
scoring/call_scorer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call Summary Scorer using Groq LLM (llama-3.1-8b-instant).
Evaluates sales call summary text, updates call_score and last_call_summary,
blends it with profile and WhatsApp scores, and updates Supabase.
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
from conversation_store import _get_supabase
from scoring.engine import calculate_blended_score, classify_lead_type

logger = logging.getLogger("scoring.call_scorer")

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _get_groq_client() -> Groq:
    load_dotenv(override=True)
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    return Groq(api_key=api_key)


def _analyze_call_summary(summary: str) -> dict:
    """
    Call Groq to analyze the call summary and classify call intent/score.
    """
    client = _get_groq_client()
    prompt = f"""You are a sales call analyst for Star Health Insurance.
Analyze this call summary and return JSON:
{{
  "intent": "hot|warm|cold",
  "score": <integer 0-100>,
  "reason": "<one line explanation>",
  "next_action": "<recommended next step for sales team>"
}}
Call Summary: "{summary}"
Return ONLY valid JSON, no markdown.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        # Clean any markdown code wrappers
        if content.startswith("```"):
            content = re.sub(r"^```(json)?\n", "", content, flags=re.IGNORECASE)
            content = re.sub(r"\n```$", "", content)
        
        return json.loads(content.strip())
    except Exception as exc:
        logger.error(f"Groq call analysis failed: {exc}")
        return {
            "intent": "warm",
            "score": 50,
            "reason": f"Fallback: analysis error ({exc})",
            "next_action": "Follow up with customer via call or WhatsApp"
        }


def _update_call_score_sync(lead_id: str, call_summary: str) -> dict | None:
    """
    Synchronous DB operations for updating lead score based on call summary.
    Called inside a thread pool to avoid blocking the event loop.
    """
    try:
        db = _get_supabase()
        
        # 1. Fetch current lead state
        res_lead = db.table("leads").select("*").eq("id", lead_id).execute()
        rows = res_lead.data or []
        if not rows:
            logger.warning(f"No lead found with ID: {lead_id} for call scoring.")
            return None
        
        lead = rows[0]
        
        # 2. Call Groq for call summary analysis
        analysis = _analyze_call_summary(call_summary)
        intent = analysis.get("intent", "warm")
        call_score = int(analysis.get("score", 50))
        reason = analysis.get("reason", "Analyzed call summary")
        next_action = analysis.get("next_action", "Follow up")
        
        # 3. Calculate WhatsApp behavior score component if applicable
        whatsapp_behavior_score = None
        last_intent = lead.get("last_whatsapp_intent")
        if last_intent is not None:
            cumulative_delta = lead.get("whatsapp_score_delta") or 0
            whatsapp_behavior_score = max(0.0, min(100.0, 70.0 + cumulative_delta))
        
        # 4. Blend scores
        profile_score = lead.get("profile_score")
        if profile_score is None:
            # Fallback to current ai_rank_score if profile_score was not backfilled
            profile_score = lead.get("ai_rank_score")
            
        final_score = calculate_blended_score(
            profile_score = profile_score,
            whatsapp_behavior = whatsapp_behavior_score,
            call_signal = call_score
        )
        
        new_lead_type = classify_lead_type(final_score)
        
        # 5. Update leads table in Supabase
        update_payload = {
            "call_score": call_score,
            "last_call_summary": call_summary,
            "ai_rank_score": int(final_score),
            "lead_type": new_lead_type,
            "score_updated_at": _now_iso(),
            "ai_rank_explanation": f"Call rating: {call_score} ({intent}). Rationale: {reason}. Recommended action: {next_action}."
        }
        
        res_update = db.table("leads").update(update_payload).eq("id", lead_id).select().execute()
        updated_rows = res_update.data or []
        if updated_rows:
            logger.info(
                f"Successfully updated lead_id={lead_id} call score: score={int(final_score)} ({new_lead_type})"
            )
            return updated_rows[0]
            
        return None
        
    except Exception as e:
        logger.error(f"Failed to update lead score from call summary for lead_id {lead_id}: {e}", exc_info=True)
        return None


async def update_lead_score_from_call(lead_id: str, call_summary: str) -> dict | None:
    """
    Asynchronously update the lead score from a sales call summary.
    """
    logger.info(f"Triggering call score update for lead_id {lead_id}...")
    return await asyncio.to_thread(_update_call_score_sync, lead_id, call_summary)
