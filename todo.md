# Project Roadmap: Advanced Requirements

To achieve the maximum grade (Level E) based on `Project 2 LLM-Powered Intelligent Agents.pdf`, the following features must be implemented.

## 🚀 Priority 1: Advanced Ingestion (Images)
**Requirement:** "Accept scanned images or handwritten documents (JPG, PNG)... Apply OCR."
- [ ] **Update `backend/rag/ingest.py`**:
    - [ ] Add `.jpg`, `.jpeg`, and `.png` to `SUPPORTED_EXTENSIONS`.
    - [ ] Implement `extract_text_from_image` function using `pytesseract` and `PIL`.
    - [ ] Ensure `tesseract-ocr` is installed in the Dockerfile (Already done, just need the Python code).
- [ ] **Update Frontend Upload**:
    - [ ] Allow image file selection in the file picker.

## 📊 Priority 2: Structured Data Export
**Requirement:** "The extracted structured information is exported to CSV, Excel, or DOCX files."
- [ ] **Create Export Endpoint (`backend/fastapi_server.py`)**:
    - [ ] Create `/export_csv` route.
    - [ ] Write a specific LLM prompt to force JSON output (e.g., "Extract maintenance specs into JSON").
    - [ ] Convert JSON to CSV using Python's `csv` module.
    - [ ] Return `StreamingResponse` (downloadable file).
- [ ] **Frontend Integration**:
    - [ ] Add a "Download Report" button near the chat window.

## 🧠 Priority 3: UI & Source Citations
**Requirement:** "Answers display correctly... Source attribution."
- [ ] **Frontend Source Rendering**:
    - [ ] Update `ChatMessage.tsx` to check for a `sources` prop.
    - [ ] Render a small "Sources" section at the bottom of the assistant's message bubble (Filename + Page Number).

## 📝 Priority 4: Documentation (The Report)
**Requirement:** "Provide step-by-step instructions... details about system setup."
- [ ] **Create `README.md`**:
    - [ ] **Prerequisites**: List Docker, Git, and how to get an API Key.
    - [ ] **Installation**: `git clone`, `.env` creation, `docker-compose up`.
    - [ ] **Usage Guide**: How to upload a file and ask a question.
    - [ ] **Technical Stack**: Diagram or list (FastAPI, Next.js, ChromaDB, Gemini).
- [ ] **Write Project Report**:
    - [ ] Discuss why RAG (Retrieval Augmented Generation) was used.
    - [ ] Discuss challenges (e.g., getting Docker networking to work).

## ✅ Completed Features
- [x] Basic Web Interface (Next.js)
- [x] File Ingestion (PDF, DOCX, TXT)
- [x] Vector Database Implementation (ChromaDB)
- [x] LLM Integration (Gemini 2.5)
- [x] Conversational Memory (Follow-up questions)
- [x] Docker Containerization