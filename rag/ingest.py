#!/usr/bin/env python3
"""
rag/ingest.py - Multi-format document ingestion with duplicate prevention

Supported Formats:
 - PDF (with OCR fallback)
 - DOC/DOCX (Microsoft Word)
 - TXT (Plain text)
 - RTF (Rich Text Format)
 - ODT (OpenDocument Text)
 - HTML/HTM (Web pages)
 - MD (Markdown)

Features:
 - Hash-based duplicate detection
 - OCR for scanned PDFs
 - Automatic format detection
 - Safe, deterministic ingestion
"""

import argparse
import os
import json
import time
import glob
import logging
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

# PDF processing
try:
    import fitz  # pymupdf
except Exception:
    fitz = None

try:
    from pdf2image import convert_from_path
    import pytesseract
except Exception:
    convert_from_path = None
    pytesseract = None

# Document processing
try:
    import docx  # python-docx for .docx files
except Exception:
    docx = None

try:
    from docx2txt import process as docx2txt_process  # Alternative for .doc files
except Exception:
    docx2txt_process = None

try:
    import pypandoc  # For .doc, .rtf, .odt conversion
except Exception:
    pypandoc = None

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
except Exception:
    chromadb = None

# Optional token estimator
try:
    import tiktoken
except Exception:
    tiktoken = None

from tqdm import tqdm

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Supported file extensions
# -------------------------
SUPPORTED_EXTENSIONS = {
    '.pdf': 'PDF',
    '.doc': 'DOC',
    '.docx': 'DOCX',
    '.txt': 'TXT',
    '.rtf': 'RTF',
    '.odt': 'ODT',
    '.html': 'HTML',
    '.htm': 'HTML',
    '.md': 'MARKDOWN',
}

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


def get_file_type(filepath: str) -> Optional[str]:
    """Get file type from extension."""
    ext = Path(filepath).suffix.lower()
    return SUPPORTED_EXTENSIONS.get(ext)


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF with OCR fallback for scanned pages."""
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed. Install with: pip install pymupdf")
    
    logger.info(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    pages = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text").strip()

        # OCR fallback if page has no text
        if not text and convert_from_path and pytesseract:
            logger.info(f"  Page {i+1}: No text found, using OCR...")
            try:
                images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
                text = pytesseract.image_to_string(images[0]).strip()
            except Exception as e:
                logger.warning(f"  OCR failed for page {i+1}: {e}")

        # Truncate very long pages
        if len(text) > 500_000:
            text = text[:500_000]
            logger.warning(f"  Page {i+1}: Text truncated to 500k characters")

        pages.append({"page": i + 1, "text": text})

    doc.close()
    logger.info(f"Extracted {len(pages)} pages from PDF")
    return pages


def extract_text_from_docx(docx_path: str) -> List[Dict[str, Any]]:
    """Extract text from DOCX file."""
    if docx is None:
        raise RuntimeError("python-docx not installed. Install with: pip install python-docx")
    
    logger.info(f"Opening DOCX: {docx_path}")
    doc = docx.Document(docx_path)
    
    # Extract all paragraphs
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    
    # Combine into single document (DOCX doesn't have clear pages)
    full_text = "\n\n".join(paragraphs)
    
    # Return as single "page"
    pages = [{"page": 1, "text": full_text}]
    logger.info(f"Extracted text from DOCX ({len(full_text)} characters)")
    return pages


def extract_text_from_doc_with_pandoc(doc_path: str) -> List[Dict[str, Any]]:
    """Extract text from DOC using pypandoc."""
    if pypandoc is None:
        raise RuntimeError("pypandoc not installed. Install with: pip install pypandoc")
    
    logger.info(f"Opening DOC with pypandoc: {doc_path}")
    try:
        text = pypandoc.convert_file(doc_path, 'plain', format='doc')
        pages = [{"page": 1, "text": text.strip()}]
        logger.info(f"Extracted text from DOC ({len(text)} characters)")
        return pages
    except Exception as e:
        logger.error(f"Failed to convert DOC file: {e}")
        raise


def extract_text_from_txt(txt_path: str) -> List[Dict[str, Any]]:
    """Extract text from plain text file."""
    logger.info(f"Opening TXT: {txt_path}")
    
    # Try multiple encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    text = None
    
    for encoding in encodings:
        try:
            with open(txt_path, 'r', encoding=encoding) as f:
                text = f.read().strip()
            logger.info(f"Successfully read with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if text is None:
        raise RuntimeError(f"Could not decode text file with any encoding: {encodings}")
    
    pages = [{"page": 1, "text": text}]
    logger.info(f"Extracted text from TXT ({len(text)} characters)")
    return pages


def extract_text_from_rtf(rtf_path: str) -> List[Dict[str, Any]]:
    """Extract text from RTF using pypandoc."""
    if pypandoc is None:
        raise RuntimeError("pypandoc not installed. Install with: pip install pypandoc")
    
    logger.info(f"Opening RTF with pypandoc: {rtf_path}")
    try:
        text = pypandoc.convert_file(rtf_path, 'plain', format='rtf')
        pages = [{"page": 1, "text": text.strip()}]
        logger.info(f"Extracted text from RTF ({len(text)} characters)")
        return pages
    except Exception as e:
        logger.error(f"Failed to convert RTF file: {e}")
        raise


def extract_text_from_html(html_path: str) -> List[Dict[str, Any]]:
    """Extract text from HTML file."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
    
    logger.info(f"Opening HTML: {html_path}")
    
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text
    text = soup.get_text(separator='\n', strip=True)
    
    pages = [{"page": 1, "text": text}]
    logger.info(f"Extracted text from HTML ({len(text)} characters)")
    return pages


def extract_text_from_document(doc_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from any supported document format.
    Returns list of pages with text content.
    """
    file_type = get_file_type(doc_path)
    
    if file_type is None:
        ext = Path(doc_path).suffix
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(SUPPORTED_EXTENSIONS.keys())}")
    
    logger.info(f"Detected file type: {file_type}")
    
    try:
        if file_type == 'PDF':
            return extract_text_from_pdf(doc_path)
        
        elif file_type == 'DOCX':
            return extract_text_from_docx(doc_path)
        
        elif file_type == 'DOC':
            # Try pypandoc first, fall back to docx2txt
            if pypandoc:
                return extract_text_from_doc_with_pandoc(doc_path)
            elif docx2txt_process:
                logger.info(f"Opening DOC with docx2txt: {doc_path}")
                text = docx2txt_process(doc_path)
                return [{"page": 1, "text": text.strip()}]
            else:
                raise RuntimeError("No DOC converter available. Install pypandoc or docx2txt")
        
        elif file_type == 'TXT' or file_type == 'MARKDOWN':
            return extract_text_from_txt(doc_path)
        
        elif file_type == 'RTF' or file_type == 'ODT':
            if pypandoc:
                return extract_text_from_rtf(doc_path)
            else:
                raise RuntimeError("pypandoc required for RTF/ODT files. Install with: pip install pypandoc")
        
        elif file_type == 'HTML':
            return extract_text_from_html(doc_path)
        
        else:
            raise ValueError(f"Handler not implemented for {file_type}")
    
    except Exception as e:
        logger.error(f"Failed to extract text from {doc_path}: {e}")
        raise


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
    doc_uid: str,
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
                    "id": f"{doc_uid}_chunk_{chunk_id}",
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
                    "id": f"{doc_uid}_chunk_{chunk_id}",
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
        # First check if collection has any documents
        count = collection.count()
        if count == 0:
            return False
            
        result = collection.get(
            where={"file_hash": file_hash},
            limit=1,
            include=["metadatas"]
        )
        
        found = len(result.get("ids", [])) > 0
        
        if found:
            logger.info(f"Found existing document with hash: {file_hash}")
        else:
            logger.debug(f"No existing document found with hash: {file_hash}")
            
        return found
    except Exception as e:
        logger.warning(f"Error checking for existing document (hash: {file_hash}): {e}")
        return False


# -------------------------
# Main Ingestion Function
# -------------------------

def ingest_documents(
    doc_paths: List[str],
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
    Ingest documents of various formats into ChromaDB with duplicate detection.
    
    Supported formats: PDF, DOC, DOCX, TXT, RTF, ODT, HTML, MD
    
    Args:
        doc_paths: List of document file paths
        force_reingest: If True, re-ingest even if document already exists
    """

    logger.info(f"Ingesting {len(doc_paths)} documents")

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

    manifest = {
        "collection": collection_name,
        "persist_dir": persist_dir,
        "sources": [],
        "errors": []
    }
    total_chunks_ingested = 0
    skipped_files = []
    error_files = []

    for doc_path in doc_paths:
        try:
            logger.info(f"Processing document: {doc_path}")

            # Check file type
            file_type = get_file_type(doc_path)
            if file_type is None:
                logger.warning(f"Skipping unsupported file type: {doc_path}")
                error_files.append({"file": doc_path, "error": "Unsupported file type"})
                continue

            # Compute file hash for duplicate detection
            file_hash = compute_file_hash(doc_path)
            logger.info(f"Computed file hash: {file_hash}")
            
            # Check if already ingested
            if not force_reingest and check_document_exists(collection, file_hash):
                logger.info(f"Document already exists (hash: {file_hash}), skipping: {doc_path}")
                skipped_files.append(doc_path)
                continue
            
            logger.info(f"Document is new or force_reingest=True, proceeding with ingestion")

            # Unique ID per document using hash
            base = Path(doc_path).stem
            doc_uid = f"{base}_{file_hash}"

            # Extract text based on file type
            pages = extract_text_from_document(doc_path)

            # Check if any text was extracted
            total_text = sum(len(p["text"]) for p in pages)
            if total_text == 0:
                logger.warning(f"No text extracted from {doc_path}, skipping")
                error_files.append({"file": doc_path, "error": "No text extracted"})
                continue

            chunks = chunk_text(
                pages,
                doc_uid=doc_uid,
                chunk_size=chunk_size,
                overlap=overlap,
                use_token=use_token_chunks
            )

            BATCH = 64
            ids, metas, docs = [], [], []
            doc_manifest = {
                "document": doc_path,
                "file_type": file_type,
                "uid": doc_uid,
                "file_hash": file_hash,
                "num_pages": len(pages),
                "chunks": []
            }

            for c in chunks:
                meta = {
                    "document_id": doc_uid,
                    "file_hash": file_hash,
                    "file_type": file_type,
                    "source_filename": os.path.basename(doc_path),
                    "source": os.path.basename(doc_path),
                    "absolute_path": os.path.abspath(doc_path),
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
            
        except Exception as e:
            logger.error(f"Error processing {doc_path}: {e}")
            error_files.append({"file": doc_path, "error": str(e)})
            continue

    manifest["total_chunks"] = total_chunks_ingested
    manifest["skipped_files"] = skipped_files
    manifest["errors"] = error_files
    manifest_path = os.path.join(persist_dir, f"{collection_name}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Ingestion completed. Total chunks: {total_chunks_ingested}, Skipped: {len(skipped_files)}, Errors: {len(error_files)}")
    return manifest


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Ingest documents (PDF, DOC, DOCX, TXT, etc.) into ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported file formats:
{chr(10).join(f"  {ext}: {name}" for ext, name in SUPPORTED_EXTENSIONS.items())}

Examples:
  # Ingest all PDFs in a folder
  python ingest.py --docs ./manuals/*.pdf
  
  # Ingest mixed document types
  python ingest.py --docs manual.pdf guide.docx notes.txt
  
  # Force re-ingestion
  python ingest.py --docs manual.pdf --force_reingest
"""
    )
    p.add_argument("--docs", nargs="+", required=True, help="Document paths (supports wildcards)")
    p.add_argument("--persist_dir", default="./vector_store", help="Vector store directory")
    p.add_argument("--collection", default="car_manuals", help="Collection name")
    p.add_argument("--provider", choices=["openai", "local"], default="local", help="Embedding provider")
    p.add_argument("--openai_key", default=None, help="OpenAI API key")
    p.add_argument("--openai_model", default="text-embedding-3-large")
    p.add_argument("--sentence_model", default="all-MiniLM-L6-v2")
    p.add_argument("--chunk_size", type=int, default=1200, help="Chunk size in characters")
    p.add_argument("--overlap", type=int, default=200, help="Overlap between chunks")
    p.add_argument("--use_token_chunks", action="store_true", help="Use token-based chunking")
    p.add_argument("--force_reingest", action="store_true", 
                   help="Force re-ingestion even if document already exists")
    return p.parse_args()


def main():
    args = parse_args()

    # Expand wildcards and collect all files
    docs = []
    for path in args.docs:
        if "*" in path:
            docs.extend(glob.glob(path))
        else:
            docs.append(path)
    
    # Filter to only existing files
    docs = [d for d in docs if os.path.isfile(d)]
    
    if not docs:
        logger.error("No valid document files found!")
        return

    logger.info(f"Found {len(docs)} documents to process")

    ingest_documents(
        doc_paths=docs,
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