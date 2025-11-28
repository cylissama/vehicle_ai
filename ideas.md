Commands

curl -X POST "http://localhost:8000/ingest" \
  -F "/Users/cylis/Projects/vehicle_maintenance/pdfs/wikiHonda_CR-V.pdf" \
  -F "collection_name=my_car_manuals"

simpler

curl -X POST "http://localhost:8000/ingest" \
  -F "file=@/Users/cylis/Projects/vehicle_maintenance/pdfs/wikiHonda_CR-V.pdf"




use a RAG (Retreival Augmented QA) System

embeds the pdfs into a vector database then uses an LLM o top to get the data and answer questions

Architecture

- PDF to Text extraction
    - pymupdf
- Chunck the Text
- Embed each Chunck
    - openai text embedding-3-large
- store embeddings in a vector DB 
    - chromaDB
- retreival-augmented pipeline
- use structured extraction on top of Retreival


PDF → Text Extraction → Chunking → Embedding → Vector DB → LLM → Next.js UI

Core Components
	1.	Backend API
	•	Handles:
	•	Uploading PDF
	•	Extracting + chunking text
	•	Generating embeddings
	•	Storing in vector DB
	•	Querying with RAG pipeline
	2.	Vector Database
	•	Options:
	•	ChromaDB (local, easiest)
	•	Qdrant (simple + powerful)
	•	Pinecone (best hosted)
	3.	LLM API
	•	You can choose:
	•	OpenAI GPT-4.1
	•	OR local LLaMA 3.1 8B/70B via LM Studio or Ollama
	4.	Web UI (Next.js App Router)
	•	Search bar
	•	Clean results panel
	•	Placeholder for future:
	•	Maintenance reminders
	•	Car mileage tracking
	•	User accounts
	•	Notifications

