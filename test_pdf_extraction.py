#!/usr/bin/env python3
"""
Test PDF text extraction to see if OCR is working properly
"""
import sys
import fitz  # pymupdf
from pdf2image import convert_from_path
import pytesseract

def test_pdf_extraction(pdf_path: str, sample_pages: list = [1, 2, 3]):
    """
    Test text extraction from specific pages of a PDF
    Shows both direct extraction and OCR results
    """
    print("=" * 70)
    print(f"PDF EXTRACTION TEST: {pdf_path}")
    print("=" * 70)
    
    try:
        doc = fitz.open(pdf_path)
        print(f"\n📄 PDF Info:")
        print(f"  • Total pages: {doc.page_count}")
        print(f"  • File size: {doc.metadata.get('format', 'unknown')}")
        
        for page_num in sample_pages:
            if page_num > doc.page_count:
                continue
                
            print(f"\n{'=' * 70}")
            print(f"PAGE {page_num}")
            print(f"{'=' * 70}")
            
            # Try direct text extraction
            page = doc.load_page(page_num - 1)  # 0-indexed
            text = page.get_text("text").strip()
            
            print(f"\n📝 Direct text extraction (pymupdf):")
            print(f"  • Length: {len(text)} chars")
            if text:
                print(f"  • First 300 chars:")
                print(f"    {text[:300]}")
                print(f"  • Quality: ✅ GOOD (text layer exists)")
            else:
                print(f"  • Quality: ⚠️  EMPTY (likely scanned image)")
            
            # Try OCR if text is empty or very short
            if len(text) < 100:
                print(f"\n🔍 Attempting OCR (Tesseract)...")
                try:
                    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0]).strip()
                        print(f"  • OCR Length: {len(ocr_text)} chars")
                        if ocr_text:
                            print(f"  • First 300 chars:")
                            print(f"    {ocr_text[:300]}")
                            print(f"  • Quality: ✅ OCR SUCCESSFUL")
                        else:
                            print(f"  • Quality: ❌ OCR FAILED (no text extracted)")
                except Exception as e:
                    print(f"  • OCR Error: {e}")
            
            # Check for images on page
            image_list = page.get_images()
            print(f"\n🖼️  Images on page: {len(image_list)}")
            if len(image_list) > 0 and len(text) < 100:
                print(f"  ⚠️  Warning: Page has images but little text - likely needs OCR")
        
        doc.close()
        
        print(f"\n{'=' * 70}")
        print("✅ Extraction test complete")
        print(f"{'=' * 70}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pdf_extraction.py <path_to_pdf> [page1,page2,page3]")
        print("\nExample:")
        print("  python test_pdf_extraction.py ./pdfs/wikiHonda_CR-V.pdf 1,2,3")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Parse page numbers if provided
    if len(sys.argv) > 2:
        pages = [int(p) for p in sys.argv[2].split(",")]
    else:
        pages = [1, 2, 3]
    
    test_pdf_extraction(pdf_path, pages)