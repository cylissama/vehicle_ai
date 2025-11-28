#!/usr/bin/env python3
"""
rag/ingest.py - Improved version with bug fixes and duplicate prevention

Fixes Applied:
 - Correctly create pdf_uid inside ingest_pdfs and pass to chunk_text
 - Removed undefined variable "pdf" inside chunk_text
 - No rewriting of existing Chroma data
 - Guaranteed unique chunk IDs with hash-based approach
 - Duplicate detection to prevent re-ingesting same content
 - All ingestion steps safe & deterministic
"""

import argparse
import os
import json
import time
import glob
import logging
import hashlib
from typing import List, Dict, Any

import fitz  # pymupdf
from tqdm import tqdm
from pdf2image import convert_from_path
import pytesseract

# Embedding libs
try:
    import openai
except Exception:
    openai = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Vector DB
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None

# Optional token estimator
try:
    import tiktoken
except Exception:
    tiktoken = None

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Utilities
# -------------------------

def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of file for duplicate detection."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:16]  # Use first 16 chars for brevity


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    logger.info(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    pages = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text").strip()

        if not text:
            images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
            text = pytesseract.image_to_string(images[0]).strip()

        if len(text) > 500_000:
            text = text[:500_000]

        pages.append({"page": i + 1, "text": text})

    doc.close()
    return pages


def simple_token_count(text: str, model_name: str = "gpt-4o") -> int:
    if tiktoken:
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    else:
        return max(1, len(text) // 4)


def chunk_text(
    pages: List[Dict[str, Any]],
    pdf_uid: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    use_token: bool = False,
    token_model: str = "gpt-4o"
) -> List[Dict[str, Any]]:
    """Produce chunks safely without overwriting existing IDs."""

    logger.info(f"Starting chunking: chunk_size={chunk_size}, overlap={overlap}, use_token={use_token}")

    chunks = []
    chunk_id = 0

    for p in tqdm(pages, desc="Chunking pages", unit="page"):
        text = p["text"].strip()
        if not text:
            continue

        step = chunk_size - overlap

        if use_token:
            words = text.split()
            starts = list(range(0, len(words), step))
            for start in starts:
                window_words = words[start:start + chunk_size]
                chunk_text_block = " ".join(window_words)
                token_len = simple_token_count(chunk_text_block, token_model)

                chunks.append({
                    "id": f"{pdf_uid}_chunk_{chunk_id}",
                    "page": p["page"],
                    "chunk_index": chunk_id,
                    "text": chunk_text_block,
                    "char_len": len(chunk_text_block),
                    "token_len": token_len,
                    "source_page_text_preview": chunk_text_block[:250]
                })
                chunk_id += 1

        else:
            starts = list(range(0, len(text), step))
            for start in starts:
                end = min(len(text), start + chunk_size)
                chunk_text_block = text[start:end]
                token_len = simple_token_count(chunk_text_block, token_model)

                chunks.append({
                    "id": f"{pdf_uid}_chunk_{chunk_id}",
                    "page": p["page"],
                    "chunk_index": chunk_id,
                    "text": chunk_text_block,
                    "char_len": len(chunk_text_block),
                    "token_len": token_len,
                    "source_page_text_preview": chunk_text_block[:250]
                })
                chunk_id += 1

    logger.info(f"Finished chunking: {len(chunks)} chunks created")
    return chunks


# -------------------------
# Embedding Providers
# -------------------------

class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        if openai is None:
            raise RuntimeError("OpenAI not installed.")
        openai.api_key = api_key
        self.model = model

    def embed_texts(self, texts: List[str]):
        out = []
        BATCH = 16
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            resp = openai.Embedding.create(model=self.model, input=batch)
            for item in resp["data"]:
                out.append(item["embedding"])
            time.sleep(0.1)
        return out


class LocalSentenceEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed.")
        logger.info(f"Loading SentenceTransformer: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()


# -------------------------
# Chroma Helpers
# -------------------------

def get_chroma_client(persist_dir: str = "./vector_store"):
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def check_document_exists(collection, file_hash: str) -> bool:
    """Check if a document with this hash already exists in the collection."""
    try:
        result = collection.get(
            where={"file_hash": file_hash},
            limit=1
        )
        return len(result.get("ids", [])) > 0
    except Exception as e:
        logger.warning(f"Error checking for existing document: {e}")
        return False


# -------------------------
# Main Ingestion Function
# -------------------------

def ingest_pdfs(
    pdf_paths: List[str],
    persist_dir: str,
    collection_name: str,
    provider: str = "openai",
    openai_key: str = None,
    openai_model: str = "text-embedding-3-large",
    sentence_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    overlap: int = 200,
    use_token_chunks: bool = False,
    force_reingest: bool = False
):
    """
    Ingest PDFs into ChromaDB with duplicate detection.
    
    Args:
        force_reingest: If True, re-ingest even if document already exists
    """

    logger.info(f"Ingesting {len(pdf_paths)} PDFs")

    # Embedding Provider
    if provider == "openai":
        embedder = OpenAIEmbedder(openai_key, openai_model)
    else:
        embedder = LocalSentenceEmbedder(sentence_model)

    client = get_chroma_client(persist_dir)

    try:
        collection = client.get_collection(collection_name)
        logger.info(f"Using existing collection: {collection_name}")
    except Exception:
        collection = client.create_collection(collection_name)
        logger.info(f"Created new collection: {collection_name}")

    manifest = {"collection": collection_name, "persist_dir": persist_dir, "sources": []}
    total_chunks_ingested = 0
    skipped_files = []

    for pdf in pdf_paths:
        logger.info(f"Processing PDF: {pdf}")

        # Compute file hash for duplicate detection
        file_hash = compute_file_hash(pdf)
        
        # Check if already ingested
        if not force_reingest and check_document_exists(collection, file_hash):
            logger.info(f"Document already exists (hash: {file_hash}), skipping: {pdf}")
            skipped_files.append(pdf)
            continue

        # Unique ID per PDF ingest using hash instead of timestamp
        base = os.path.splitext(os.path.basename(pdf))[0]
        pdf_uid = f"{base}_{file_hash}"

        pages = extract_text_from_pdf(pdf)

        chunks = chunk_text(
            pages,
            pdf_uid=pdf_uid,
            chunk_size=chunk_size,
            overlap=overlap,
            use_token=use_token_chunks
        )

        BATCH = 64
        ids, metas, docs = [], [], []
        doc_manifest = {
            "pdf": pdf,
            "uid": pdf_uid,
            "file_hash": file_hash,
            "num_pages": len(pages),
            "chunks": []
        }

        for c in chunks:
            meta = {
                "document_id": pdf_uid,
                "file_hash": file_hash,  # Add hash to metadata
                "source_filename": os.path.basename(pdf),
                "source": os.path.basename(pdf),  # Add this for compatibility
                "absolute_path": os.path.abspath(pdf),
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "char_len": c["char_len"],
                "token_len": c["token_len"],
                "preview": c["source_page_text_preview"]
            }

            ids.append(c["id"])
            metas.append(meta)
            docs.append(c["text"])

            doc_manifest["chunks"].append({
                "id": c["id"],
                "page": c["page"],
                "char_len": c["char_len"]
            })

            if len(ids) >= BATCH:
                embeddings = embedder.embed_texts(docs)
                collection.add(ids=ids, metadatas=metas, documents=docs, embeddings=embeddings)
                total_chunks_ingested += len(ids)
                ids, metas, docs = [], [], []

        if ids:
            embeddings = embedder.embed_texts(docs)
            collection.add(ids=ids, metadatas=metas, documents=docs, embeddings=embeddings)
            total_chunks_ingested += len(ids)

        manifest["sources"].append(doc_manifest)

    manifest["total_chunks"] = total_chunks_ingested
    manifest["skipped_files"] = skipped_files
    manifest_path = os.path.join(persist_dir, f"{collection_name}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Ingestion completed. Total chunks: {total_chunks_ingested}, Skipped: {len(skipped_files)}")
    return manifest


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    p.add_argument("--pdfs", nargs="+", required=True)
    p.add_argument("--persist_dir", default="./vector_store")
    p.add_argument("--collection", default="car_manuals")
    p.add_argument("--provider", choices=["openai", "local"], default="local")
    p.add_argument("--openai_key", default=None)
    p.add_argument("--openai_model", default="text-embedding-3-large")
    p.add_argument("--sentence_model", default="all-MiniLM-L6-v2")
    p.add_argument("--chunk_size", type=int, default=1200)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--use_token_chunks", action="store_true")
    p.add_argument("--force_reingest", action="store_true", 
                   help="Force re-ingestion even if document already exists")
    return p.parse_args()


def main():
    args = parse_args()

    pdfs = []
    for path in args.pdfs:
        if "*" in path:
            pdfs.extend(glob.glob(path))
        else:
            pdfs.append(path)
    pdfs = [p for p in pdfs if os.path.isfile(p)]

    ingest_pdfs(
        pdf_paths=pdfs,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        provider=args.provider,
        openai_key=args.openai_key,
        openai_model=args.openai_model,
        sentence_model=args.sentence_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        use_token_chunks=args.use_token_chunks,
        force_reingest=args.force_reingest
    )


if __name__ == "__main__":
    main()