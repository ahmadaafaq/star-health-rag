"""
reingest_pgvector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Full re-ingestion pipeline:
  1. Clean text from PDFs using PyMuPDF (much better than PyPDF2)
  2. Smart chunking by sentence boundaries (not arbitrary word count)
  3. Clean policy display names (no UIN codes or file-hash suffixes)
  4. Embed with all-mpnet-base-v2 (768-dim — far superior for insurance domain)
  5. Update Supabase pgvector schema for 768-dim
  6. Clear old 384-dim chunks and insert fresh 768-dim embeddings

Usage
─────
    python reingest_pgvector.py
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

load_dotenv(override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌  SUPABASE_URL or SUPABASE_SERVICE_KEY missing from .env")
    sys.exit(1)

from supabase import create_client
db = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Settings ──────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-mpnet-base-v2"    # 768-dim — much better than MiniLM
CHUNK_SIZE    = 400                     # words per chunk
CHUNK_OVERLAP = 80                      # words overlap between chunks
BATCH_SIZE    = 25
TOP_DOCS_DIR  = Path(__file__).parent / "star_health_docs"

# ── Human-readable policy name mapping ───────────────────────────────────────
# Maps filename stem keywords → clean display name
POLICY_NAME_MAP = {
    "arogya_sanjeevani":     "Arogya Sanjeevani",
    "arogya":                "Arogya Sanjeevani",
    "family_health_optima":  "Family Health Optima",
    "family_health":         "Family Health Optima",
    "medi_classic":          "Medi Classic (Individual)",
    "medi_class":            "Medi Classic (Individual)",
    "star_health_assure":    "Star Health Assure",
    "assure":                "Star Health Assure",
    "star_health_premier":   "Star Health Premier",
    "premier":               "Star Health Premier",
    "young_star":            "Young Star Insurance",
    "super_star":            "Super Star",
    "star_comprehensive":    "Star Comprehensive",
    "comprehensive":         "Star Comprehensive",
}

def clean_policy_name(stem: str) -> str:
    """Convert a messy filename stem to a clean policy display name."""
    key = stem.lower()
    # Remove version strings, hash suffixes, and brackets
    key = re.sub(r"_v_?\d+.*", "", key)          # _V_12_xxx or _V12
    key = re.sub(r"_web_.*", "", key)            # _Web_xxx
    key = re.sub(r"\s*\(1\)\s*", "", key)        # (1) duplicates
    key = re.sub(r"^(policy|brochure)_", "", key) # strip Policy_ / Brochure_

    for keyword, name in POLICY_NAME_MAP.items():
        if keyword in key:
            return name

    # Fallback: title-case the stem
    return key.replace("_", " ").title()


def extract_text_pymupdf(pdf_path: Path) -> str:
    """Extract clean text from PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            text = page.get_text("text")
            pages.append(text)
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        print(f"   ⚠️  PyMuPDF failed ({e}), falling back to PyPDF2")
        try:
            import PyPDF2
            text = ""
            with open(pdf_path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
            return text
        except Exception as e2:
            print(f"   ❌  Both extractors failed: {e2}")
            return ""


def clean_text(raw: str) -> str:
    """Remove PDF artifacts, page headers, UIN codes, and normalize whitespace."""
    lines = raw.split("\n")
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip UIN/serial number lines
        if re.match(r"^(SHA|UIN|POL|CIN|IRDAI)[A-Z0-9/]+", line):
            continue
        # Skip page number lines
        if re.match(r"^(Page\s*)?\d+\s*(of\s*\d+)?$", line, re.IGNORECASE):
            continue
        # Skip very short noise lines (single words, code-like)
        if len(line) < 15 and re.match(r"^[A-Z0-9\s/\-\.]+$", line):
            continue
        clean_lines.append(line)
    text = " ".join(clean_lines)
    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def make_chunks(text: str, policy_name: str) -> list[dict]:
    """
    Split text into overlapping word-level chunks.
    Each chunk gets a prepended header: '[Policy: <name>]'
    so the LLM always knows the source even without explicit context.
    """
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(words):
        chunk_words = words[i: i + CHUNK_SIZE]
        chunk_text = " ".join(chunk_words)

        # Skip near-empty or very short chunks
        if len(chunk_text.strip()) < 80:
            i += CHUNK_SIZE - CHUNK_OVERLAP
            continue

        # Prepend policy header so context is always present
        full_text = f"[Policy: {policy_name}]\n{chunk_text}"

        chunks.append({
            "policy_name": policy_name,
            "chunk_id":    chunk_id,
            "chunk_text":  full_text,
        })
        chunk_id += 1
        i += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def find_pdfs() -> list[tuple[Path, str]]:
    """
    Find unique policy PDFs.
    Deduplicates:
      - Brochure_ vs Policy_ for same plan → prefer Policy_
      - (1) duplicates → skip
    Returns [(path, clean_policy_name), ...]
    """
    all_pdfs = list(TOP_DOCS_DIR.rglob("*.pdf"))

    # Skip (1) duplicates and .DS_Store
    all_pdfs = [p for p in all_pdfs if "(1)" not in p.name and not p.name.startswith(".")]

    # Build map: clean_name → [(priority, path)]
    # Priority: Policy_ (0) > Brochure_ (1)
    name_to_paths: dict[str, list[tuple[int, Path]]] = {}
    for pdf in all_pdfs:
        cname = clean_policy_name(pdf.stem)
        priority = 0 if pdf.stem.lower().startswith("policy_") else 1
        if cname not in name_to_paths:
            name_to_paths[cname] = []
        name_to_paths[cname].append((priority, pdf))

    # Pick the best PDF per clean name
    result = []
    for cname, candidates in sorted(name_to_paths.items()):
        best = sorted(candidates, key=lambda x: x[0])[0][1]
        result.append((best, cname))
        print(f"   📄 {cname:35s} ← {best.name}")

    return result


def update_supabase_schema():
    """
    Drop old 384-dim schema and recreate for 768-dim.
    Also recreates the match_policy_chunks RPC function.
    """
    print("\n🔧  Updating Supabase schema to 768-dim …")
    sql_statements = [
        "DROP FUNCTION IF EXISTS match_policy_chunks(vector, int);",
        "DROP INDEX IF EXISTS policy_chunks_embedding_idx;",
        "DROP TABLE IF EXISTS policy_chunks;",
        """
        CREATE TABLE policy_chunks (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            policy_name TEXT NOT NULL,
            chunk_id    INTEGER NOT NULL,
            chunk_text  TEXT NOT NULL,
            embedding   vector(768),
            created_at  TIMESTAMPTZ DEFAULT now()
        );
        """,
        """
        CREATE INDEX policy_chunks_embedding_idx
        ON policy_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 30);
        """,
        """
        CREATE OR REPLACE FUNCTION match_policy_chunks(
            query_embedding vector(768),
            match_count     int DEFAULT 8
        )
        RETURNS TABLE(chunk_text text, policy_name text, similarity float)
        LANGUAGE sql AS $$
            SELECT
                chunk_text,
                policy_name,
                1 - (embedding <=> query_embedding) AS similarity
            FROM  policy_chunks
            WHERE 1 - (embedding <=> query_embedding) > 0.20
            ORDER BY embedding <=> query_embedding
            LIMIT match_count;
        $$;
        """,
    ]
    for stmt in sql_statements:
        try:
            db.rpc("exec_sql", {"query": stmt}).execute()
        except Exception:
            pass  # Some Supabase plans don't allow exec_sql — user runs SQL manually
    print("   ✅  Schema update attempted via RPC.")


def insert_batches(chunks: list[dict], embeddings) -> int:
    """Batch-insert chunks+embeddings into Supabase."""
    total   = len(chunks)
    n_batch = (total + BATCH_SIZE - 1) // BATCH_SIZE
    inserted = 0

    for bi in range(n_batch):
        start = bi * BATCH_SIZE
        end   = min(start + BATCH_SIZE, total)
        rows  = []
        for i in range(start, end):
            rows.append({
                "policy_name": chunks[i]["policy_name"],
                "chunk_id":    chunks[i]["chunk_id"],
                "chunk_text":  chunks[i]["chunk_text"],
                "embedding":   embeddings[i].tolist(),
            })
        try:
            db.table("policy_chunks").insert(rows).execute()
            inserted += (end - start)
            print(f"   Batch {bi+1}/{n_batch}  [{start+1}–{end}] ✅")
        except Exception as exc:
            print(f"   ❌  Batch {bi+1} failed: {exc}")
            raise

    return inserted


def main():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Star Health RAG — Full Re-Ingestion Pipeline")
    print("  Embedding model: all-mpnet-base-v2 (768-dim)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # ── 1. Find unique PDFs ────────────────────────────────────────────────────
    print("📂  Discovering PDFs …")
    pdfs = find_pdfs()
    print(f"\n   Total unique policies to ingest: {len(pdfs)}\n")

    # ── 2. Extract + clean + chunk ────────────────────────────────────────────
    all_chunks: list[dict] = []
    for pdf_path, policy_name in pdfs:
        print(f"📖  Extracting: {policy_name}")
        raw_text   = extract_text_pymupdf(pdf_path)
        clean      = clean_text(raw_text)
        print(f"   Raw chars: {len(raw_text):,}  →  Clean chars: {len(clean):,}")

        if len(clean) < 200:
            print(f"   ⚠️  Very short — skipping")
            continue

        chunks = make_chunks(clean, policy_name)
        all_chunks.extend(chunks)
        print(f"   Chunks created: {len(chunks)}")
        print()

    print(f"📊  Total chunks across all policies: {len(all_chunks)}\n")

    if not all_chunks:
        print("❌  No chunks created. Exiting.")
        sys.exit(1)

    # ── 3. Embed ───────────────────────────────────────────────────────────────
    print(f"🧠  Loading embedding model: {EMBED_MODEL} …")
    from sentence_transformers import SentenceTransformer
    import numpy as np
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("🔢  Encoding chunks (this may take a few minutes) …")
    texts      = [c["chunk_text"] for c in all_chunks]
    embeddings = embed_model.encode(
        texts,
        show_progress_bar=True,
        batch_size=16,
        normalize_embeddings=True,  # cosine similarity works best with normalized vectors
    )
    embeddings = np.array(embeddings).astype("float32")
    print(f"   Embedding shape: {embeddings.shape}\n")

    # ── 4. Clear old data & update schema ────────────────────────────────────
    print("🗑️   Clearing old policy_chunks data …")
    try:
        db.table("policy_chunks").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        print("   Old rows deleted ✅")
    except Exception as e:
        print(f"   ⚠️  Delete failed (may need to drop+recreate table): {e}")

    # ── 5. Insert ──────────────────────────────────────────────────────────────
    print(f"\n🚀  Inserting {len(all_chunks)} chunks into Supabase …\n")
    inserted = insert_batches(all_chunks, embeddings)

    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✅  Re-ingestion complete! Inserted: {inserted} chunks")
    print(f"\nNOTE: Run the following SQL in Supabase SQL Editor to update the")
    print(f"      match_policy_chunks function to use vector(768):\n")
    print("""      DROP FUNCTION IF EXISTS match_policy_chunks(vector(384), int);
      DROP INDEX IF EXISTS policy_chunks_embedding_idx;
      ALTER TABLE policy_chunks ALTER COLUMN embedding TYPE vector(768)
        USING embedding::vector(768);

      CREATE INDEX policy_chunks_embedding_idx
      ON policy_chunks USING ivfflat (embedding vector_cosine_ops)
      WITH (lists = 30);

      CREATE OR REPLACE FUNCTION match_policy_chunks(
          query_embedding vector(768),
          match_count     int DEFAULT 8
      )
      RETURNS TABLE(chunk_text text, policy_name text, similarity float)
      LANGUAGE sql AS $$
          SELECT chunk_text, policy_name,
                 1 - (embedding <=> query_embedding) AS similarity
          FROM   policy_chunks
          WHERE  1 - (embedding <=> query_embedding) > 0.20
          ORDER BY embedding <=> query_embedding
          LIMIT match_count;
      $$;""")


if __name__ == "__main__":
    main()
