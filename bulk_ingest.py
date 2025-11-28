#!/usr/bin/env python3
"""
Bulk ingest all PDFs from a directory using the robust rag/ingest.py module
"""
import os
import glob
from dotenv import load_dotenv
from rag.ingest import ingest_pdfs

# Load environment variables
load_dotenv()

# Configuration
PDF_DIR = "/Users/cylis/Projects/vehicle_maintenance/pdfs"
VECTOR_DIR = os.getenv("VECTOR_DIR", "./vector_store")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
SENTENCE_MODEL = os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2")

def main():
    print("=" * 60)
    print("BULK PDF INGESTION SCRIPT")
    print("=" * 60)
    
    # Find all PDFs
    pdf_pattern = os.path.join(PDF_DIR, "*.pdf")
    pdf_files = sorted(glob.glob(pdf_pattern))
    
    if not pdf_files:
        print(f"❌ No PDF files found in: {PDF_DIR}")
        return
    
    print(f"\n📁 Found {len(pdf_files)} PDF files:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {os.path.basename(pdf)}")
    
    print(f"\n⚙️  Configuration:")
    print(f"  • Vector store: {VECTOR_DIR}")
    print(f"  • Collection: {COLLECTION_NAME}")
    print(f"  • Embedding model: {SENTENCE_MODEL}")
    print(f"  • Chunk size: 1200 chars")
    print(f"  • Overlap: 200 chars")
    
    confirm = input(f"\n🚀 Proceed with ingestion? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return
    
    print(f"\n{'=' * 60}")
    print("STARTING INGESTION")
    print(f"{'=' * 60}\n")
    
    # Run ingestion using the robust rag/ingest.py module
    manifest = ingest_pdfs(
        pdf_paths=pdf_files,
        persist_dir=VECTOR_DIR,
        collection_name=COLLECTION_NAME,
        provider="local",
        sentence_model=SENTENCE_MODEL,
        chunk_size=1200,
        overlap=200,
        use_token_chunks=False
    )
    
    print(f"\n{'=' * 60}")
    print("✅ INGESTION COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\n📊 Summary:")
    print(f"  • Total chunks ingested: {manifest['total_chunks']}")
    print(f"  • PDFs processed: {len(manifest['sources'])}")
    print(f"  • Manifest saved: {VECTOR_DIR}/{COLLECTION_NAME}_manifest.json")
    
    print(f"\n📝 Per-file breakdown:")
    for source in manifest['sources']:
        pdf_name = os.path.basename(source['pdf'])
        print(f"  • {pdf_name}: {source['num_pages']} pages → {len(source['chunks'])} chunks")
    
    print(f"\n🎉 All PDFs successfully ingested into collection '{COLLECTION_NAME}'")
    print(f"You can now query them via your FastAPI server!\n")

if __name__ == "__main__":
    main()