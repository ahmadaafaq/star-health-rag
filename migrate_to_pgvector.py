"""
migrate_to_pgvector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One-time migration: reads vectors directly from the FAISS index (no
re-computation) and uploads every chunk + its 384-dim embedding to the
Supabase policy_chunks table.

Prerequisites
─────────────
1. Run sql/pgvector_setup.sql in the Supabase SQL Editor first.
2. Ensure .env has SUPABASE_URL and SUPABASE_SERVICE_KEY set.

Usage
─────
    python migrate_to_pgvector.py
"""

from __future__ import annotations

import os
import pickle
import sys
from typing import List

import faiss
import numpy as np
from dotenv import load_dotenv

# ── Load env ────────────────────────────────────────────────────────────────
load_dotenv(override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌  SUPABASE_URL or SUPABASE_SERVICE_KEY missing from .env")
    sys.exit(1)

# ── Supabase client ──────────────────────────────────────────────────────────
from supabase import create_client
db = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Paths ────────────────────────────────────────────────────────────────────
INDEX_DIR  = os.path.join(os.path.dirname(__file__), "faiss_index")
FAISS_PATH = os.path.join(INDEX_DIR, "index.faiss")
CHUNKS_PKL = os.path.join(INDEX_DIR, "chunks.pkl")
BATCH_SIZE = 50


def load_data() -> tuple[List[dict], np.ndarray]:
    """Load chunks metadata and extract all vectors from FAISS without re-encoding."""
    print("📂  Loading FAISS index and chunks.pkl …")

    if not os.path.exists(FAISS_PATH):
        print(f"❌  FAISS index not found at: {FAISS_PATH}")
        sys.exit(1)
    if not os.path.exists(CHUNKS_PKL):
        print(f"❌  chunks.pkl not found at: {CHUNKS_PKL}")
        sys.exit(1)

    index = faiss.read_index(FAISS_PATH)
    with open(CHUNKS_PKL, "rb") as fh:
        chunks: List[dict] = pickle.load(fh)

    print(f"✅  Loaded {index.ntotal} vectors (dim={index.d}) and {len(chunks)} chunk metadata records.")

    if index.ntotal != len(chunks):
        print(f"⚠️   Vector count ({index.ntotal}) ≠ chunk count ({len(chunks)}). Will use min of both.")

    # Extract all raw vectors — no re-encoding needed
    n = min(index.ntotal, len(chunks))
    vectors: np.ndarray = index.reconstruct_n(0, n)   # shape (n, 384)
    return chunks[:n], vectors


def check_existing() -> int:
    """Return count of rows already in policy_chunks."""
    try:
        result = db.table("policy_chunks").select("id", count="exact").execute()
        return result.count or 0
    except Exception as exc:
        print(f"⚠️   Could not check existing rows: {exc}")
        return 0


def insert_batches(chunks: List[dict], vectors: np.ndarray) -> int:
    """Insert all chunks+embeddings in batches of BATCH_SIZE."""
    total   = len(chunks)
    n_batch = (total + BATCH_SIZE - 1) // BATCH_SIZE
    inserted = 0

    for batch_idx in range(n_batch):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, total)

        rows = []
        for i in range(start, end):
            chunk = chunks[i]
            rows.append({
                "policy_name": chunk["policy"],
                "chunk_id":    chunk.get("chunk_id", i),
                "chunk_text":  chunk["text"],
                "embedding":   vectors[i].tolist(),   # list[float] → stored as vector(384)
            })

        try:
            db.table("policy_chunks").insert(rows).execute()
            inserted += (end - start)
            print(f"   Inserting batch {batch_idx + 1}/{n_batch}  "
                  f"[rows {start + 1}–{end}] ✅")
        except Exception as exc:
            print(f"   ❌  Batch {batch_idx + 1} failed: {exc}")
            raise

    return inserted


def main() -> None:
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Star Health pgvector Migration Tool")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # 1. Check for duplicate data
    existing = check_existing()
    if existing > 0:
        print(f"⚠️   policy_chunks already contains {existing} row(s).")
        answer = input("   Proceed anyway and insert more rows? [y/N]: ").strip().lower()
        if answer != "y":
            print("Migration cancelled. No data inserted.")
            sys.exit(0)

    # 2. Load FAISS data
    chunks, vectors = load_data()

    # 3. Insert
    print(f"\n🚀  Inserting {len(chunks)} chunks in batches of {BATCH_SIZE} …\n")
    inserted = insert_batches(chunks, vectors)

    print(f"\n✅  Migration complete! Total inserted: {inserted} chunks")
    print("   You can now verify with:")
    print("   SELECT COUNT(*) FROM policy_chunks;  -- in Supabase SQL Editor\n")


if __name__ == "__main__":
    main()
