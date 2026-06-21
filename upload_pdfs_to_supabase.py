"""
upload_pdfs_to_supabase.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One-time script: uploads the 7 policy PDFs to a public Supabase Storage bucket
called 'policy-pdfs' so Twilio can fetch them as WhatsApp media attachments.

Prerequisites
─────────────
1. SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env
2. Run this ONCE — it skips files that are already uploaded.

Usage
─────
    python upload_pdfs_to_supabase.py
"""

from __future__ import annotations

import os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌  SUPABASE_URL or SUPABASE_SERVICE_KEY missing from .env")
    sys.exit(1)

from supabase import create_client
db = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "policy-pdfs"
POLICIES_DIR = Path(__file__).parent / "star_health_docs" / "policies"

# Map plan ID → canonical PDF filename stem (case-insensitive prefix match)
PLAN_PDF_MAP = {
    "arogya-sanjeevani":    "Policy_Arogya_Sanjeevani",
    "family-health-optima": "Policy_Family_Health_Optima",
    "medi-classic":         "Policy_Medi_Classic",
    "star-assure":          "Policy_Star_Health_Assure_Insurance_Policy_V_9_c53663e68a.pdf",
    "star-premier":         "Policy_Star_Health_Premier",
    "super-star":           "Policy_Super_Star_V_3_80e5dd8988.pdf",
    "young-star":           "Policy_Young_Star_Insurance_Policy_V_12_59f47a25f5.pdf",
    "star-comprehensive":   "Brochure_Star_Comprehensive_Insurance_Policy_V_15_Web_633bcfcaaf.pdf",
}

BROCHURES_DIR = Path(__file__).parent / "star_health_docs" / "brochure"

def find_pdf(stem_or_file: str) -> Path | None:
    """Find a PDF in the policies or brochure directory matching the given stem."""
    # Check policies dir first (full filename or prefix)
    direct = POLICIES_DIR / stem_or_file
    if direct.exists():
        return direct
    for f in sorted(POLICIES_DIR.glob("*.pdf")):
        if f.name.startswith(stem_or_file) and "(1)" not in f.name:
            return f
    # Fall back to brochure dir
    direct_b = BROCHURES_DIR / stem_or_file
    if direct_b.exists():
        return direct_b
    for f in sorted(BROCHURES_DIR.glob("*.pdf")):
        if f.name.startswith(stem_or_file) and "(1)" not in f.name:
            return f
    return None


def ensure_bucket():
    """Create the bucket if it doesn't exist."""
    try:
        existing = db.storage.list_buckets()
        names = [b.name for b in existing]
        if BUCKET_NAME not in names:
            db.storage.create_bucket(BUCKET_NAME, options={"public": True})
            print(f"✅  Created public bucket: {BUCKET_NAME}")
        else:
            print(f"ℹ️   Bucket '{BUCKET_NAME}' already exists.")
    except Exception as e:
        print(f"⚠️   Could not check/create bucket: {e}")


def upload_pdf(plan_id: str, stem_or_file: str) -> str | None:
    """Upload one PDF and return its public URL."""
    pdf_path = find_pdf(stem_or_file)
    if not pdf_path:
        print(f"   ❌  PDF not found for plan '{plan_id}' (pattern: {stem_or_file})")
        return None

    object_name = f"{plan_id}.pdf"

    try:
        with open(pdf_path, "rb") as fh:
            content = fh.read()
        db.storage.from_(BUCKET_NAME).upload(
            path=object_name,
            file=content,
            file_options={"content-type": "application/pdf", "upsert": "true"},
        )
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{object_name}"
        print(f"   ✅  Uploaded: {pdf_path.name} → {object_name}")
        print(f"       URL: {public_url}")
        return public_url
    except Exception as e:
        print(f"   ❌  Upload failed for '{plan_id}': {e}")
        return None


def main():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Star Health Policy PDF Upload Tool")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    ensure_bucket()
    print()

    results = {}
    for plan_id, stem in PLAN_PDF_MAP.items():
        print(f"📄  Processing: {plan_id}")
        url = upload_pdf(plan_id, stem)
        if url:
            results[plan_id] = url
        print()

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✅  Done! {len(results)}/{len(PLAN_PDF_MAP)} PDFs uploaded.\n")
    print("Copy these URLs into POLICY_PDF_MAP in api.py:")
    for plan_id, url in results.items():
        print(f'    "{plan_id}": "{url}",')


if __name__ == "__main__":
    main()
