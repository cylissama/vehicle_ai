#!/bin/bash

PDF_DIR="/Users/cylis/Projects/vehicle_maintenance/pdfs"
API_URL="http://localhost:8000/ingest"

echo "Starting PDF ingestion..."

for pdf in "$PDF_DIR"/*.pdf; do
    if [ -f "$pdf" ]; then
        echo "Uploading: $(basename "$pdf")"
        curl -X POST "$API_URL" \
          -F "file=@$pdf" \
          -F "collection_name=car_references"
        echo -e "\n---"
    fi
done

echo "All PDFs uploaded!"