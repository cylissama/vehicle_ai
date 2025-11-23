#!/usr/bin/env python3
"""
rag/ingest.py - Improved version

Usage examples:
  python rag/ingest.py --pdfs manuals/owners_manual.pdf manuals/service_manual.pdf \
      --persist_dir ./vector_store --provider openai --openai_key YOUR_OPENAI_KEY

  python rag/ingest.py --pdfs manuals/*.pdf --persist_dir ./vector_store \
      --provider local --sentence_model all-MiniLM-L6-v2

Features:
 - Extracts text from PDFs (pymupdf + OCR fallback)
 - Splits into overlapping chunks (efficient, fast)
 - Generates embeddings (OpenAI or sentence-transformers)
 - Stores embeddings in ChromaDB (persistent)
 - Logs progress at every step
"""
import argparse
import os
import json
import time
import glob
import logging
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
def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    logger.info(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    pages = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text").strip()

        if not text:
            # Page likely scanned → run OCR
            images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
            text = pytesseract.image_to_string(images[0]).strip()

        if len(text) > 500_000:
            logger.warning(f"Page {i+1} very long ({len(text)} chars), truncating to 500k chars")
            text = text[:500_000]

        logger.info(f"Page {i+1} text length: {len(text)} chars")
        pages.append({"page": i+1, "text": text})

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

def chunk_text(pages: List[Dict[str, Any]], chunk_size: int = 1000, overlap: int = 200,
               use_token: bool = False, token_model: str = "gpt-4o") -> List[Dict[str, Any]]:
    """
    Efficient chunking with progress bars.
    Returns list of chunks: {id, page, chunk_index, text, char_len, token_len}
    """
    logger.info(f"Starting chunking: chunk_size={chunk_size}, overlap={overlap}, use_token={use_token}")
    chunks = []
    chunk_id = 0

    for p in tqdm(pages, desc="Chunking pages", unit="page"):
        text = p["text"].strip()
        if not text:
            continue

        # Determine sliding window step
        step = chunk_size - overlap

        if use_token:
            words = text.split()
            starts = list(range(0, len(words), step))
            for start in tqdm(starts, desc=f"Page {p['page']} chunks", leave=False, unit="chunk"):
                window_words = words[start:start+chunk_size]
                chunk_text_block = " ".join(window_words)
                token_len = simple_token_count(chunk_text_block, token_model)
                chunks.append({
                    "id": f"chunk_{chunk_id}",
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
            for start in tqdm(starts, desc=f"Page {p['page']} chunks", leave=False, unit="chunk"):
                end = min(len(text), start + chunk_size)
                chunk_text_block = text[start:end]
                token_len = simple_token_count(chunk_text_block, token_model)
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "page": p["page"],
                    "chunk_index": chunk_id,
                    "text": chunk_text_block,
                    "char_len": len(chunk_text_block),
                    "token_len": token_len,
                    "source_page_text_preview": chunk_text_block[:250]
                })
                chunk_id += 1

        if chunk_id % 50 == 0:
            logger.info(f"  {chunk_id} chunks created so far...")

    logger.info(f"Finished chunking: {len(chunks)} chunks created")
    return chunks

# -------------------------
# Embedding providers
# -------------------------
class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        if openai is None:
            raise RuntimeError("openai library not installed. pip install openai")
        openai.api_key = api_key
        self.model = model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        out = []
        BATCH = 16
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            resp = openai.Embedding.create(model=self.model, input=batch)
            for item in resp["data"]:
                out.append(item["embedding"])
            time.sleep(0.1)
        return out

class LocalSentenceEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
        logger.info(f"Loading SentenceTransformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded local embedding model: {model_name}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

# -------------------------
# Chroma helper
# -------------------------
def get_chroma_client(persist_dir: str = "./vector_store"):
    if chromadb is None:
        raise RuntimeError("chromadb not installed. pip install chromadb")
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    logger.info(f"Chroma client initialized at: {persist_dir}")
    return client

# -------------------------
# Main ingestion
# -------------------------
def ingest_pdfs(pdf_paths: List[str], persist_dir: str, collection_name: str,
                provider: str = "openai", openai_key: str = None, openai_model: str = "text-embedding-3-large",
                sentence_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, overlap: int = 200,
                use_token_chunks: bool = False) -> Dict[str, Any]:

    logger.info(f"Starting ingestion: {len(pdf_paths)} PDFs, provider={provider}")

    # Initialize embedder
    if provider == "openai":
        if not openai_key:
            raise ValueError("OpenAI selected but no API key provided")
        embedder = OpenAIEmbedder(openai_key, openai_model)
        embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_key, model_name=openai_model)
    elif provider == "local":
        embedder = LocalSentenceEmbedder(sentence_model)
        embedding_fn = None
    else:
        raise ValueError("Unsupported provider. Choose 'openai' or 'local'.")

    client = get_chroma_client(persist_dir)

    # Create/get collection
    try:
        collection = client.get_collection(name=collection_name)
        logger.info(f"Using existing Chroma collection: {collection_name}")
    except Exception:
        collection = client.create_collection(name=collection_name)
        logger.info(f"Created new Chroma collection: {collection_name}")

    manifest = {"collection": collection_name, "persist_dir": persist_dir, "sources": []}
    total_chunks_ingested = 0

    for pdf in pdf_paths:
        logger.info(f"Processing PDF: {pdf}")
        pages = extract_text_from_pdf(pdf)
        chunks = chunk_text(pages, chunk_size=chunk_size, overlap=overlap, use_token=use_token_chunks)
        logger.info(f"Extracted {len(pages)} pages -> {len(chunks)} chunks")

        # Batch embedding & upsert
        BATCH = 64
        ids, metadatas, texts_for_embed = [], [], []
        doc_manifest = {"pdf": pdf, "num_pages": len(pages), "chunks": []}

        for c in chunks:
            chunk_meta = {
                "source": os.path.basename(pdf),
                "absolute_path": os.path.abspath(pdf),
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "char_len": c["char_len"],
                "token_len": c["token_len"],
                "preview": c["source_page_text_preview"]
            }
            ids.append(c["id"])
            metadatas.append(chunk_meta)
            texts_for_embed.append(c["text"])
            doc_manifest["chunks"].append({"id": c["id"], "page": c["page"], "char_len": c["char_len"]})

            if len(ids) >= BATCH:
                embeddings = embedder.embed_texts(texts_for_embed)
                collection.add(ids=ids, metadatas=metadatas, documents=texts_for_embed, embeddings=embeddings)
                total_chunks_ingested += len(ids)
                ids, metadatas, texts_for_embed = [], [], []

        if ids:
            embeddings = embedder.embed_texts(texts_for_embed)
            collection.add(ids=ids, metadatas=metadatas, documents=texts_for_embed, embeddings=embeddings)
            total_chunks_ingested += len(ids)

        manifest["sources"].append(doc_manifest)
        logger.info(f"Ingested {len(doc_manifest['chunks'])} chunks from {os.path.basename(pdf)}")

    manifest["total_chunks"] = total_chunks_ingested
    manifest_path = os.path.join(persist_dir, f"{collection_name}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Ingestion complete. Total chunks: {total_chunks_ingested}")
    logger.info(f"Chroma persisted to: {persist_dir}")
    logger.info(f"Manifest saved to: {manifest_path}")

    return manifest

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Ingest PDFs into a Chroma vector store with embeddings.")
    p.add_argument("--pdfs", nargs="+", required=True, help="Paths to PDF files (supports globs).")
    p.add_argument("--persist_dir", default="./vector_store", help="Chroma persist directory.")
    p.add_argument("--collection", default="car_manuals", help="Chroma collection name.")
    p.add_argument("--provider", choices=["openai", "local"], default="local", help="Embedding provider.")
    p.add_argument("--openai_key", default=None, help="OpenAI API key (if provider=openai).")
    p.add_argument("--openai_model", default="text-embedding-3-large", help="OpenAI embedding model.")
    p.add_argument("--sentence_model", default="all-MiniLM-L6-v2", help="SentenceTransformers model if provider=local.")
    p.add_argument("--chunk_size", type=int, default=1200, help="Chunk size (chars or tokens).")
    p.add_argument("--overlap", type=int, default=200, help="Overlap between chunks.")
    p.add_argument("--use_token_chunks", action="store_true", help="Use token-based chunking.")
    return p.parse_args()

def main():
    args = parse_args()

    pdf_paths = []
    for p in args.pdfs:
        if any(ch in p for ch in ["*", "?"]):
            pdf_paths.extend(sorted(glob.glob(p)))
        else:
            pdf_paths.append(p)
    pdf_paths = [p for p in pdf_paths if os.path.isfile(p)]
    if not pdf_paths:
        raise SystemExit("No valid PDF paths found.")

    ingest_pdfs(
        pdf_paths=pdf_paths,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        provider=args.provider,
        openai_key=args.openai_key,
        openai_model=args.openai_model,
        sentence_model=args.sentence_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        use_token_chunks=args.use_token_chunks
    )

if __name__ == "__main__":
    main()