import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import PyPDF2

load_dotenv()

# Settings
DOCS_PATH = "star_health_docs"
INDEX_PATH = "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

print("Model loading...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def make_chunks(text, policy_name):
    chunks = []
    words = text.split()
    i = 0
    while i < len(words):
        chunk_words = words[i:i + CHUNK_SIZE]
        chunk_text = " ".join(chunk_words)
        if len(chunk_text.strip()) > 50:
            chunks.append({
                "text": chunk_text,
                "policy": policy_name,
                "chunk_id": len(chunks)
            })
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def ingest():
    all_chunks = []
    docs_path = Path(DOCS_PATH)
    
    pdf_files = list(docs_path.rglob("*.pdf"))
    
    if not pdf_files:
        print(f"no pdf in {DOCS_PATH}")
        return
    
    print(f"\n{len(pdf_files)} PDFs exists:")
    
    for pdf_path in pdf_files:
        policy_name = pdf_path.stem
        print(f"  Processing: {policy_name}")
        
        text = extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            print(f"  Warning: not from {policy_name} ")
            continue
        
        chunks = make_chunks(text, policy_name)
        all_chunks.extend(chunks)
        print(f"  {len(chunks)} chunks ready")
    
    if not all_chunks:
        print("no chunks !")
        return
    
    print(f"\nTotal {len(all_chunks)} chunks — embeddings process...")
    
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    os.makedirs(INDEX_PATH, exist_ok=True)
    faiss.write_index(index, f"{INDEX_PATH}/index.faiss")
    
    with open(f"{INDEX_PATH}/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    
    print(f"\nDone! faiss_index/ save ")
    print(f"Total chunks indexed: {len(all_chunks)}")

if __name__ == "__main__":
    ingest()