# fastapi_server.py - Multi-format document support
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import logging
import uvicorn
from dotenv import load_dotenv
from pathlib import Path
from pydantic import HttpUrl
import requests
from bs4 import BeautifulSoup
import hashlib
import time


import chromadb
from rag.ingest import (
    ingest_documents, 
    LocalSentenceEmbedder, 
    compute_file_hash, 
    check_document_exists,
    SUPPORTED_EXTENSIONS,
    chunk_text,
)

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("fastapi_server")

# -------------------------
# Config
# -------------------------
VECTOR_DIR = os.getenv("VECTOR_DIR", "./vector_store")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_car_manuals")
SENTENCE_MODEL = os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "6"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables!")
    raise ValueError("GEMINI_API_KEY is required")

# -------------------------
# Initialize Chroma
# -------------------------
os.makedirs(VECTOR_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=VECTOR_DIR)
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    logger.info(f"Loaded existing collection: {COLLECTION_NAME}")
except Exception:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    logger.info(f"Created collection: {COLLECTION_NAME}")

# -------------------------
# Initialize local embedder
# -------------------------
local_embedder = LocalSentenceEmbedder(SENTENCE_MODEL)

# -------------------------
# Initialize Gemini client
# -------------------------
import google.generativeai as genai

logger.info(f"Initializing Google Gemini with model: {GEMINI_MODEL}")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="Vehicle Maintenance RAG API - Multi-Format Support")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Pydantic models
# -------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class DebugQueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 6

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = TOP_K
    use_llm: Optional[bool] = True
    history: List[ChatMessage] = []

class QueryResponse(BaseModel):
    answer: str
    source_chunks: List[dict]

# -------------------------
# Helper function
# -------------------------
def is_supported_file(filename: str) -> bool:
    """Check if file extension is supported."""
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS

def get_file_type_name(filename: str) -> str:
    """Get human-readable file type name."""
    ext = Path(filename).suffix.lower()
    return SUPPORTED_EXTENSIONS.get(ext, "Unknown")

# -------------------------
# Routes
# -------------------------
@app.get("/status")
def status():
    return {
        "ok": True,
        "collection": COLLECTION_NAME,
        "vector_dir": os.path.abspath(VECTOR_DIR),
        "model": GEMINI_MODEL,
        "total_chunks": collection.count(),
        "supported_formats": list(SUPPORTED_EXTENSIONS.keys())
    }

@app.get("/supported_formats")
def supported_formats():
    """List all supported file formats."""
    return {
        "formats": [
            {"extension": ext, "type": name}
            for ext, name in SUPPORTED_EXTENSIONS.items()
        ],
        "upload_instructions": {
            "pdf": "PDFs with OCR support for scanned documents",
            "docx": "Microsoft Word documents (.docx)",
            "doc": "Legacy Word documents (.doc) - requires pandoc",
            "txt": "Plain text files",
            "rtf": "Rich Text Format - requires pandoc",
            "odt": "OpenDocument Text - requires pandoc",
            "html": "HTML/HTM web pages",
            "md": "Markdown files"
        }
    }

@app.get("/collections")
def list_collections():
    cols = chroma_client.list_collections()
    return {"collections": [col.name for col in cols]}

@app.get("/collection_peek")
def peek_collection():
    """See what's in the collection"""
    result = collection.peek(limit=10)
    return {
        "collection_name": COLLECTION_NAME,
        "count": collection.count(),
        "sample_docs": result.get("documents", [])[:3],
        "sample_metadata": result.get("metadatas", [])[:3]
    }

@app.get("/collection_stats")
def collection_stats():
    """Get detailed statistics about the collection"""
    count = collection.count()
    
    if count > 0:
        result = collection.get(limit=count, include=["metadatas"])
        metadatas = result.get("metadatas", [])
        
        # Count chunks per source and file type
        source_counts = {}
        file_type_counts = {}
        file_hashes = set()
        
        for meta in metadatas:
            source = meta.get("source_filename", meta.get("source", "unknown"))
            file_type = meta.get("file_type", "unknown")
            file_hash = meta.get("file_hash")
            
            if file_hash:
                file_hashes.add(file_hash)
            
            source_counts[source] = source_counts.get(source, 0) + 1
            file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
        
        return {
            "collection_name": COLLECTION_NAME,
            "total_chunks": count,
            "unique_documents": len(file_hashes),
            "file_types": file_type_counts,
            "sources": source_counts
        }
    else:
        return {
            "collection_name": COLLECTION_NAME,
            "total_chunks": 0,
            "message": "Collection is empty"
        }

@app.get("/list_documents")
def list_documents():
    """List all unique documents in the collection by file hash"""
    try:
        count = collection.count()
        if count == 0:
            return {
                "total_chunks": 0,
                "unique_documents": 0,
                "documents": []
            }
        
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])
        
        # Group by file_hash
        docs_by_hash = {}
        for meta in metadatas:
            file_hash = meta.get("file_hash", "unknown")
            if file_hash not in docs_by_hash:
                docs_by_hash[file_hash] = {
                    "file_hash": file_hash,
                    "filename": meta.get("source_filename", "unknown"),
                    "file_type": meta.get("file_type", "unknown"),
                    "path": meta.get("absolute_path", "unknown"),
                    "chunk_count": 0,
                    "pages": set()
                }
            docs_by_hash[file_hash]["chunk_count"] += 1
            docs_by_hash[file_hash]["pages"].add(meta.get("page", 0))
        
        # Convert to list and format
        documents = []
        for doc_info in docs_by_hash.values():
            doc_info["page_count"] = len(doc_info["pages"])
            doc_info["pages"] = sorted(list(doc_info["pages"]))
            documents.append(doc_info)
        
        return {
            "total_chunks": count,
            "unique_documents": len(documents),
            "documents": sorted(documents, key=lambda x: x["filename"])
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return {"error": str(e)}

@app.get("/search_by_source")
def search_by_source(source_name: str, limit: int = 10):
    """Search for chunks from a specific source file"""
    result = collection.get(
        where={"source_filename": source_name},
        limit=limit,
        include=["documents", "metadatas"]
    )
    
    return {
        "source": source_name,
        "chunks_found": len(result.get("documents", [])),
        "chunks": [
            {
                "page": meta.get("page"),
                "file_type": meta.get("file_type"),
                "full_text": doc,
                "preview": doc[:200] + "..." if len(doc) > 200 else doc,
                "char_len": meta.get("char_len")
            }
            for doc, meta in zip(result.get("documents", []), result.get("metadatas", []))
        ]
    }

@app.post("/check_duplicate")
async def check_duplicate(file: UploadFile = File(...)):
    """
    Check if a document has already been ingested without actually ingesting it.
    Returns the file hash and whether it exists in the collection.
    """
    # Check if file type is supported
    if not is_supported_file(file.filename):
        return {
            "error": f"Unsupported file type. Supported formats: {list(SUPPORTED_EXTENSIONS.keys())}"
        }
    
    # Save temporarily to compute hash
    target_dir = "./temp_checks"
    os.makedirs(target_dir, exist_ok=True)
    temp_path = os.path.join(target_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        file_hash = compute_file_hash(temp_path)
        exists = check_document_exists(collection, file_hash)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {
            "filename": file.filename,
            "file_type": get_file_type_name(file.filename),
            "file_hash": file_hash,
            "already_ingested": exists,
            "message": "Document already exists in collection" if exists else "Document is new"
        }
    except Exception as e:
        logger.error(f"Error checking duplicate: {e}")
        return {"error": str(e)}

@app.post("/debug_query")
async def debug_query(req: DebugQueryRequest):
    """Debug endpoint - shows exactly what the vector search returns"""
    question = req.question
    top_k = req.top_k
    
    logger.info(f"Debug query: {question}")
    
    q_embedding = local_embedder.embed_texts([question])[0]
    
    result = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = []
    if "documents" in result and result["documents"]:
        docs_list = result["documents"][0]
        metas_list = result.get("metadatas", [[]])[0]
        dists_list = result.get("distances", [[]])[0]
        
        for i, (doc, meta, dist) in enumerate(zip(docs_list, metas_list, dists_list)):
            chunks.append({
                "rank": i + 1,
                "distance": round(dist, 4),
                "source": meta.get("source_filename", meta.get("source", "unknown")),
                "file_type": meta.get("file_type", "unknown"),
                "page": meta.get("page", "?"),
                "char_len": meta.get("char_len", 0),
                "token_len": meta.get("token_len", 0),
                "full_text": doc,
                "preview": doc[:300] + "..." if len(doc) > 300 else doc
            })
    
    return {
        "question": question,
        "top_k": top_k,
        "chunks_retrieved": len(chunks),
        "chunks": chunks
    }

@app.post("/ingest")
async def ingest_uploaded_document(
    file: UploadFile = File(...), 
    collection_name: str = Form(None),
    force_reingest: bool = Form(False)
):
    """
    Upload and ingest a document of any supported format.
    Supports: PDF, DOC, DOCX, TXT, RTF, ODT, HTML, MD
    Includes OCR fallback for scanned PDFs.
    """
    # Check if file type is supported
    if not is_supported_file(file.filename):
        return {
            "error": f"Unsupported file type. Supported formats: {list(SUPPORTED_EXTENSIONS.keys())}",
            "supported_formats": list(SUPPORTED_EXTENSIONS.keys())
        }

    target_dir = "./uploaded_documents"
    os.makedirs(target_dir, exist_ok=True)
    dest_path = os.path.join(target_dir, file.filename)

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Saved uploaded document to: {dest_path}")

    # Determine which collection to use
    coll_name = collection_name or COLLECTION_NAME
    
    # Get or create the target collection
    try:
        target_collection = chroma_client.get_collection(name=coll_name)
    except Exception:
        target_collection = chroma_client.create_collection(name=coll_name)
        logger.info(f"Created new collection: {coll_name}")

    # Check if already ingested (unless force_reingest is True)
    if not force_reingest:
        file_hash = compute_file_hash(dest_path)
        if check_document_exists(target_collection, file_hash):
            logger.info(f"Document already ingested (hash: {file_hash}): {file.filename}")
            return {
                "ok": True,
                "already_existed": True,
                "filename": file.filename,
                "file_type": get_file_type_name(file.filename),
                "file_hash": file_hash,
                "message": "Document was already in the collection. Use force_reingest=true to re-ingest.",
                "chunks_created": 0
            }

    try:
        manifest = ingest_documents(
            doc_paths=[dest_path],
            persist_dir=VECTOR_DIR,
            collection_name=coll_name,
            provider="local",
            sentence_model=SENTENCE_MODEL,
            chunk_size=1200,
            overlap=200,
            use_token_chunks=False,
            force_reingest=force_reingest
        )
        
        # Refresh global collection reference if we used the default collection
        if coll_name == COLLECTION_NAME:
            global collection
            collection = chroma_client.get_collection(name=coll_name)
        
        return {
            "ok": True,
            "already_existed": False,
            "filename": file.filename,
            "file_type": get_file_type_name(file.filename),
            "chunks_created": manifest.get("total_chunks", 0),
            "skipped_files": manifest.get("skipped_files", []),
            "errors": manifest.get("errors", []),
            "manifest": manifest
        }
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        return {"ok": False, "error": str(e)}
    

# -------------------------
# URL Ingestion Route
# -------------------------
class URLIngestRequest(BaseModel):
    url: HttpUrl
    collection_name: Optional[str] = None
    force_reingest: bool = False


@app.post("/ingest_url")
async def ingest_url(req: URLIngestRequest):
    """
    Fetch a webpage, extract readable text, chunk it, embed it, and store in Chroma.
    Fully integrated with the existing ingest logic (duplicate detection, metadata, stats, etc.)
    """

    url = req.url
    coll_name = req.collection_name or COLLECTION_NAME
    force_reingest = req.force_reingest

    logger.info(f"Ingesting URL: {url}")

    # Get or create the collection
    try:
        target_collection = chroma_client.get_collection(name=coll_name)
    except Exception:
        logger.info(f"Creating new collection: {coll_name}")
        target_collection = chroma_client.create_collection(name=coll_name)

    # Deterministic hash for URL-based document identity
    url_hash = hashlib.sha256(str(url).encode("utf-8")).hexdigest()

    # Duplicate check
    if not force_reingest:
        if check_document_exists(target_collection, url_hash):
            logger.info(f"URL already ingested (hash={url_hash}): {url}")
            return {
                "ok": True,
                "already_existed": True,
                "url": url,
                "file_type": "url/html",
                "file_hash": url_hash,
                "message": "URL was already in the collection. Use force_reingest=true to re-ingest.",
                "chunks_created": 0
            }

    # ----------------------------------------------------------------
    # STEP 1 — Download the webpage
    # ----------------------------------------------------------------
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    # ----------------------------------------------------------------
    # STEP 2 — Extract readable text
    # ----------------------------------------------------------------
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content sections
    for tag in soup(["script", "style", "header", "footer", "nav", "noscript"]):
        tag.decompose()

    full_text = soup.get_text(separator="\n")
    full_text = "\n".join(line.strip() for line in full_text.splitlines() if line.strip())

    if len(full_text) < 40:
        raise HTTPException(status_code=400, detail="Not enough readable text extracted from webpage.")

    logger.info(f"Extracted {len(full_text)} characters of text from URL")

    # ----------------------------------------------------------------
    # STEP 3 — Create chunk structure expected by existing chunker
    # ----------------------------------------------------------------
    fake_pages = [{"page": 1, "text": full_text}]  # treat as one long "page"

    chunks = chunk_text(
        pages=fake_pages,
        doc_uid=url_hash,  # <--- Pass the hash here!
        chunk_size=1200,
        overlap=200,
        use_token=False
    )

    if not chunks:
        raise HTTPException(status_code=500, detail="Chunker returned zero chunks.")

    logger.info(f"Created {len(chunks)} chunks from URL")

    # ----------------------------------------------------------------
    # STEP 4 — Embed & store chunks in Chroma
    # ----------------------------------------------------------------
    ids, metadatas, documents = [], [], []

    timestamp = int(time.time())
    base_id = f"url_{timestamp}"

    for c in chunks:
        chunk_id = f"{base_id}_chunk_{c['chunk_index']}"
        ids.append(chunk_id)
        documents.append(c["text"])
        metadatas.append({
            "source": str(url),
            "source_filename": str(url),
            "file_type": "url/html",
            "file_hash": url_hash,
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "char_len": c["char_len"]
        })

    embeddings = local_embedder.embed_texts(documents)
    target_collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    logger.info(f"Stored {len(ids)} URL chunks in collection {coll_name}")

    # ----------------------------------------------------------------
    # STEP 5 — Return response
    # ----------------------------------------------------------------
    return {
        "ok": True,
        "url": str(url),
        "file_type": "url/html",
        "file_hash": url_hash,
        "chunks": len(chunks),
        "collection": coll_name
    }

@app.delete("/remove_document")
async def remove_document(file_hash: str):
    """Remove all chunks associated with a specific document by its file hash."""
    try:
        result = collection.get(
            where={"file_hash": file_hash},
            include=["metadatas"]
        )
        
        ids_to_delete = result.get("ids", [])
        
        if not ids_to_delete:
            return {
                "ok": False,
                "message": f"No document found with hash: {file_hash}"
            }
        
        collection.delete(ids=ids_to_delete)
        
        return {
            "ok": True,
            "file_hash": file_hash,
            "chunks_removed": len(ids_to_delete),
            "message": f"Removed {len(ids_to_delete)} chunks"
        }
    except Exception as e:
        logger.error(f"Error removing document: {e}")
        return {"ok": False, "error": str(e)}

# -------------------------
# Helper functions
# -------------------------
def build_prompt(question: str, chunks: List[dict]) -> str:
    context_lines = []
    for c in chunks:
        meta = c.get("metadata", {})
        page = meta.get("page", "?")
        source = meta.get("source_filename", meta.get("source", "manual"))
        file_type = meta.get("file_type", "")
        snippet = c.get("document")[:600] if c.get("document") else ""
        context_lines.append(f"{source} ({file_type}, page {page}):\n{snippet}")
    context_block = "\n\n---\n\n".join(context_lines)
    return context_block

def gemini_generate(question: str, context: str, history: str = "") -> str:
    """Generate answer using Google Gemini API with History and Context."""
    try:
        # We construct a prompt that includes the history (if available)
        prompt = f"""You are a helpful expert on vehicle maintenance.
Use the provided Context and Chat History to answer the user's question.

RULES:
1. Use only the information in the context or history.
2. If the user refers to "it" or "that", use the Chat History to resolve the reference.
3. If the answer is not contained in the context, say you don't know.
4. Always cite the source (filename/page) if available.

--- CHAT HISTORY ---
{history if history else "No previous history."}

--- CONTEXT ---
{context}

--- CURRENT QUESTION ---
{question}

Answer:"""

        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return f"Error generating response: {str(e)}"
# -------------------------
# Query route
# -------------------------
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question
    top_k = req.top_k or TOP_K
    use_llm = req.use_llm
    
    # 1. Format the Chat History
    # We limit to the last 6 messages to keep the token count managed
    history_text = ""
    if req.history:
        recent_history = req.history[-6:]
        for msg in recent_history:
            role = "User" if msg.role == "user" else "Assistant"
            history_text += f"{role}: {msg.content}\n"
    else:
        history_text = "No previous history."

    logger.info(f"Query received: '{question}' | top_k={top_k}, use_llm={use_llm}")

    # 2. Embed and Search (Unchanged)
    q_embedding = local_embedder.embed_texts([question])[0]

    result = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = []
    if "documents" in result and result["documents"]:
        docs_list = result["documents"][0]
        metas_list = result.get("metadatas", [[]])[0]
        dists_list = result.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs_list, metas_list, dists_list):
            docs.append({"document": doc, "metadata": meta, "distance": dist})
    else:
        logger.warning("No documents returned from Chroma query.")

    # 3. Build Context from Documents
    # (Assuming you have a build_prompt helper, or we logic it here)
    # We'll construct a raw string of the document text to pass to Gemini
    doc_text_list = [d.get("document", "") for d in docs]
    doc_context = "\n\n---\n\n".join(doc_text_list)

    if use_llm:
        # 4. Construct the Final System Prompt with History
        # We override the simple context with a rich prompt containing docs + history
        system_prompt = f"""
You are an expert vehicle maintenance assistant. 
Use the provided Context and Chat History to answer the user's question.

RULES:
1. If the answer is in the Context, use it.
2. If the user refers to "it" or "that", check the Chat History to see what they are talking about.
3. If the answer is not found, admit it.

--- CHAT HISTORY ---
{history_text}

--- DOCUMENT CONTEXT ---
{doc_context}

--- USER QUESTION ---
{question}
"""
        # We pass the full system_prompt as the 'context' to your generator
        # (Assuming gemini_generate takes (question, context) or just a prompt)
        answer = gemini_generate(question, doc_context, history_text)
    else:
        # Fallback for no-LLM mode
        snippets = []
        for d in docs:
            meta = d.get("metadata", {})
            source = meta.get("source_filename", meta.get("source", "manual"))
            page = meta.get("page", "?")
            snippet_text = d.get("document")[:600] if d.get("document") else ""
            snippets.append(f"{source} (Page {page}):\n{snippet_text}")
        answer = "Retrieved context (no LLM):\n\n" + "\n\n---\n\n".join(snippets) if snippets else "No relevant context found."

    return QueryResponse(answer=answer, source_chunks=docs)

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=5001, reload=True)