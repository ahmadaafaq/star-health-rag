-- Migration to add scoring columns to public.leads table

ALTER TABLE public.leads
  ADD COLUMN IF NOT EXISTS profile_score INTEGER,
  ADD COLUMN IF NOT EXISTS whatsapp_score_delta INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS last_whatsapp_intent TEXT,
  ADD COLUMN IF NOT EXISTS last_call_summary TEXT,
  ADD COLUMN IF NOT EXISTS call_score INTEGER,
  ADD COLUMN IF NOT EXISTS score_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- Backfill profile_score for existing leads from their current ai_rank_score
UPDATE public.leads
SET profile_score = ai_rank_score
WHERE profile_score IS NULL;
