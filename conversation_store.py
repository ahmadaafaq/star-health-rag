from __future__ import annotations

import os
import re
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

# Initialize basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("conversation_store")

load_dotenv(override=True)
# Load from web env as well if present
web_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "star-health-web", ".env")
if os.path.exists(web_env_path):
    load_dotenv(dotenv_path=web_env_path, override=False)

_RESET = "\033[0m"
_GREEN = "\033[92m"
_CYAN  = "\033[96m"
_BOLD  = "\033[1m"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalise_phone(raw: str) -> str:
    if not raw:
        return ""
    cleaned = raw.strip().replace("whatsapp:", "").strip()
    digits  = re.sub(r"[^\d]", "", cleaned)
    if digits.startswith("91") and len(digits) == 12:
        return f"+{digits}"
    if len(digits) == 10:
        return f"+91{digits}"
    if digits.startswith("0") and len(digits) == 11:
        return f"+91{digits[1:]}"
    return cleaned if cleaned.startswith("+") else f"+{digits}" if digits else cleaned


def phone_variants(phone: str) -> list[str]:
    e164   = normalise_phone(phone)
    digits = re.sub(r"[^\d]", "", e164)
    ten    = digits[-10:] if len(digits) >= 10 else digits
    return list({e164, digits, ten, f"+91{ten}"})


# ── In-memory message store ───────────────────────────────────────────────────
_memory_store: dict[str, list[dict]] = {}
_MAX_MEMORY_PER_PHONE = 100


def _memory_append(phone: str, direction: str, body: str, channel: str,
                   sid: Optional[str], message_type: Optional[str] = None) -> None:
    key = normalise_phone(phone)
    if key not in _memory_store:
        _memory_store[key] = []
    _memory_store[key].append({
        "direction":   direction,
        "body":        body,
        "channel":     channel,
        "twilio_sid":   sid,
        "message_type": message_type,
        "created_at":   _now_iso(),
    })
    if len(_memory_store[key]) > _MAX_MEMORY_PER_PHONE:
        _memory_store[key] = _memory_store[key][-_MAX_MEMORY_PER_PHONE:]


def get_conversation_memory(phone: str) -> list[dict]:
    key = normalise_phone(phone)
    return list(_memory_store.get(key, []))


# ── Supabase client — lazy, always reads from environment ─────────────────────
_supabase: Optional[object] = None


def _get_supabase():
    global _supabase
    if _supabase is None:
        load_dotenv(override=True)
        # Fallback to publishable keys if service key is not explicitly set
        url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or ""
        url = url.strip().strip('"').strip("'")
        
        key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY") or ""
        key = key.strip().strip('"').strip("'")
        
        if not url or not url.startswith("http"):
            raise RuntimeError(f"SUPABASE_URL invalid or empty: '{url}'")
        if not key:
            raise RuntimeError("SUPABASE KEY is missing")
            
        from supabase import create_client
        _supabase = create_client(url, key)
        logger.info("Supabase client initialized successfully in conversation_store.")
    return _supabase


def get_supabase_client():
    return _get_supabase()


# ── Console logging ───────────────────────────────────────────────────────────

def _print_message(direction: str, lead_phone: str, lead_name: str,
                   body: str, channel: str, sid: Optional[str] = None) -> None:
    colour   = _GREEN if direction == "inbound" else _CYAN
    arrow    = "↓ INBOUND " if direction == "inbound" else "↑ OUTBOUND"
    sid_line = f"  SID : {sid}\n" if sid else ""
    print(
        f"\n{colour}{_BOLD}{'─' * 60}\n"
        f"  {arrow}  [{channel.upper()}]  {_now_iso()}\n"
        f"{'─' * 60}{_RESET}\n"
        f"{colour}  Lead  : {lead_name} ({lead_phone})\n"
        f"  Msg   : {body[:200]}\n{sid_line}{_RESET}"
    )


# ── Lead resolution ───────────────────────────────────────────────────────────

def _resolve_lead_sync(lead_phone: str) -> Optional[str]:
    try:
        db       = _get_supabase()
        variants = phone_variants(lead_phone)
        # Query public.leads table (lowercase) instead of Lead
        result   = db.table("leads").select("id").in_("phone", variants).order("created_at", desc=True).limit(1).execute()
        rows     = result.data or []
        return rows[0]["id"] if rows else None
    except Exception as exc:
        logger.warning(f"lead_resolution_failed for phone {lead_phone}: {exc}")
        return None


def _insert_message_sync(lead_id: str, lead_phone: str, direction: str,
                         body: str, channel: str, sid: Optional[str],
                         message_type: Optional[str] = None) -> bool:
    try:
        db      = _get_supabase()
        # Query public.messages table (lowercase) with snake_case keys
        payload = {
            "lead_id":    lead_id,
            "phone":     normalise_phone(lead_phone),
            "direction": direction,
            "body":      body or "",
            "channel":   channel,
            "twilio_sid": sid,
            "created_at": _now_iso(),
        }
        if message_type:
            payload["message_type"] = message_type
        db.table("messages").insert(payload).execute()
        return True
    except Exception as exc:
        logger.warning(f"message_insert_failed for lead_id {lead_id}: {exc}")
        return False


# ── Public API ────────────────────────────────────────────────────────────────

async def record_message(
    lead_phone:   str,
    lead_name:    str,
    direction:    str,
    body:         str,
    channel:      str          = "whatsapp",
    sid:          Optional[str] = None,
    message_type: Optional[str] = None,
) -> bool:
    norm_phone = normalise_phone(lead_phone)
    _print_message(direction, norm_phone, lead_name, body, channel, sid)
    _memory_append(norm_phone, direction, body, channel, sid, message_type)
    try:
        lead_id = await asyncio.to_thread(_resolve_lead_sync, norm_phone)
        if not lead_id:
            logger.debug(f"message_memory_only_no_lead for phone: {norm_phone}")
            return False
        success = await asyncio.to_thread(
            _insert_message_sync, lead_id, norm_phone, direction, body, channel, sid, message_type
        )
        if success:
            logger.debug(f"message_saved_db for phone: {norm_phone}")
        return success
    except Exception as exc:
        logger.warning(f"record_message_db_failed for phone {norm_phone}: {exc}")
        return False


_EXCLUDED_TYPES = {"welcome", "system_notification"}
_EXCLUDED_BODY_PREFIXES = ("[DEMO]", "New Lead!", "[NEW LEAD ALERT]")


def _is_conversation_message(msg: dict) -> bool:
    msg_type = msg.get("message_type") or msg.get("messageType") or ""
    if msg_type in _EXCLUDED_TYPES:
        return False
    body = msg.get("body", "")
    if any(body.startswith(p) for p in _EXCLUDED_BODY_PREFIXES):
        return False
    if msg.get("direction") == "outbound" and len(body) > 1000:
        return False
    return True


async def get_conversation_history(lead_phone: str, limit: int = 20) -> list[dict]:
    norm_phone = normalise_phone(lead_phone)
    variants   = phone_variants(lead_phone)
    try:
        def _fetch() -> list[dict]:
            db     = _get_supabase()
            result = db.table("messages").select(
                "direction, body, channel, twilio_sid, created_at, message_type"
            ).in_("phone", variants).order("created_at", desc=True).limit(limit * 3).execute()
            rows     = result.data or []
            filtered = [r for r in rows if _is_conversation_message(r)]
            filtered.reverse()
            return filtered[-limit:]

        db_history = await asyncio.to_thread(_fetch)
        if db_history:
            logger.debug(f"history_from_db for phone {norm_phone}: count={len(db_history)}")
            return db_history
    except Exception as exc:
        logger.warning(f"history_db_failed for phone {norm_phone}: {exc}")

    mem          = get_conversation_memory(norm_phone)
    filtered_mem = [m for m in mem if _is_conversation_message(m)]
    logger.debug(f"history_from_memory for phone {norm_phone}: count={len(filtered_mem)}")
    return filtered_mem[-limit:]


async def _lookup_lead_by_phone(phone: str) -> Optional[dict]:
    try:
        db       = _get_supabase()
        variants = phone_variants(phone)
        result   = await asyncio.to_thread(
            lambda: db.table("leads").select(
                "id, name, phone, email, ai_rank_score, lead_type"
            ).in_("phone", variants).order("created_at", desc=True).limit(1).execute()
        )
        rows = result.data or []
        return rows[0] if rows else None
    except Exception as exc:
        logger.warning(f"lead_lookup_failed for phone {phone}: {exc}")
        return None
