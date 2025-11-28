# fastapi_server.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import logging
import uvicorn
from dotenv import load_dotenv

import chromadb
from rag.ingest import ingest_pdfs, LocalSentenceEmbedder, compute_file_hash, check_document_exists

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
app = FastAPI(title="Vehicle Maintenance RAG API")

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
class DebugQueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 6

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = TOP_K
    use_llm: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    source_chunks: List[dict]

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
        "total_chunks": collection.count()
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
        
        # Get all metadata
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

@app.get("/collection_stats")
def collection_stats():
    """Get detailed statistics about the collection"""
    count = collection.count()
    
    # Get all metadata to analyze sources
    if count > 0:
        result = collection.get(limit=count, include=["metadatas"])
        metadatas = result.get("metadatas", [])
        
        # Count chunks per source
        source_counts = {}
        page_counts = {}
        file_hashes = set()
        
        for meta in metadatas:
            source = meta.get("source_filename", meta.get("source", "unknown"))
            page = meta.get("page", "?")
            file_hash = meta.get("file_hash")
            
            if file_hash:
                file_hashes.add(file_hash)
            
            source_counts[source] = source_counts.get(source, 0) + 1
            page_key = f"{source}:page_{page}"
            page_counts[page_key] = page_counts.get(page_key, 0) + 1
        
        return {
            "collection_name": COLLECTION_NAME,
            "total_chunks": count,
            "unique_sources": len(source_counts),
            "unique_documents": len(file_hashes),
            "sources": source_counts,
            "top_pages": dict(sorted(page_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    else:
        return {
            "collection_name": COLLECTION_NAME,
            "total_chunks": 0,
            "message": "Collection is empty"
        }

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
    Check if a PDF has already been ingested without actually ingesting it.
    Returns the file hash and whether it exists in the collection.
    """
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files accepted."}
    
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
            "file_hash": file_hash,
            "already_ingested": exists,
            "message": "Document already exists in collection" if exists else "Document is new"
        }
    except Exception as e:
        logger.error(f"Error checking duplicate: {e}")
        return {"error": str(e)}

@app.post("/debug_query")
async def debug_query(req: DebugQueryRequest):
    """
    Debug endpoint - shows exactly what the vector search returns
    before it goes to the LLM. This helps diagnose retrieval issues.
    """
    question = req.question
    top_k = req.top_k
    
    logger.info(f"Debug query: {question}")
    
    # Get embedding for question
    q_embedding = local_embedder.embed_texts([question])[0]
    
    # Query vector DB
    result = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results for debugging
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
async def ingest_uploaded_pdf(
    file: UploadFile = File(...), 
    collection_name: str = Form(None),
    force_reingest: bool = Form(False)
):
    """
    Upload and ingest a single PDF using the robust rag/ingest.py module.
    Includes OCR fallback for scanned pages and duplicate detection.
    
    Args:
        file: The PDF file to upload
        collection_name: Optional collection name (defaults to COLLECTION_NAME)
        force_reingest: If True, re-ingest even if document already exists
    """
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF uploads accepted."}

    target_dir = "./uploaded_pdfs"
    os.makedirs(target_dir, exist_ok=True)
    dest_path = os.path.join(target_dir, file.filename)

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Saved uploaded PDF to: {dest_path}")

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
                "file_hash": file_hash,
                "message": "Document was already in the collection. Use force_reingest=true to re-ingest.",
                "chunks_created": 0
            }

    try:
        manifest = ingest_pdfs(
            pdf_paths=[dest_path],
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
            "chunks_created": manifest.get("total_chunks", 0),
            "skipped_files": manifest.get("skipped_files", []),
            "manifest": manifest
        }
    except Exception as e:
        logger.error(f"Error ingesting PDF: {e}")
        return {"ok": False, "error": str(e)}

@app.delete("/remove_document")
async def remove_document(file_hash: str):
    """
    Remove all chunks associated with a specific document by its file hash.
    """
    try:
        # Get all IDs for this document
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
        
        # Delete the chunks
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
        snippet = c.get("document")[:600] if c.get("document") else ""
        context_lines.append(f"{source} (page {page}):\n{snippet}")
    context_block = "\n\n---\n\n".join(context_lines)
    return context_block

def gemini_generate(question: str, context: str) -> str:
    """Generate answer using Google Gemini API"""
    try:
        prompt = f"""You are a helpful assistant that answers car manual questions using the provided context.
Use only the information in the context. If the answer is not contained in the context, say you don't know.
Always cite the source (filename and page number) when providing answers.

CONTEXT:
{context}

QUESTION: {question}

Provide a concise answer with sources."""
        
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

    logger.info(f"Query received: '{question}' | top_k={top_k}, use_llm={use_llm}")

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

    context = build_prompt(question, docs)

    if use_llm:
        answer = gemini_generate(question, context)
    else:
        snippets = []
        for d in docs:
            meta = d.get("metadata", {})
            source = meta.get("source_filename", meta.get("source", "manual"))
            page = meta.get("page", "?")
            snippet_text = d.get("document")[:600] if d.get("document") else ""
            snippets.append(f"{source} (page {page}):\n{snippet_text}")
        answer = "Retrieved context (no LLM):\n\n" + "\n\n---\n\n".join(snippets) if snippets else "No relevant context found in the vector store."

    return QueryResponse(answer=answer, source_chunks=docs)

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=5001, reload=True)