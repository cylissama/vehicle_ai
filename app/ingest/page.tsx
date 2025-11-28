"use client";

import { useState } from "react";

export default function IngestPage() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>("");

  const upload = async () => {
    if (!file) {
      setStatus("Please select a PDF file.");
      return;
    }

    setStatus("Uploading...");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        setStatus(`Error: ${data.error || "Upload failed"}`);
        return;
      }

      setStatus("Upload successful! Document ingested.");
    } catch (err) {
      console.error(err);
      setStatus("Upload failed. Server unreachable.");
    }
  };

  return (
    <main className="max-w-lg mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">Upload a Manual (PDF)</h1>

      <input
        type="file"
        accept="application/pdf"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
        className="mb-4"
      />

      <button
        onClick={upload}
        className="bg-blue-600 text-white px-4 py-2 rounded"
      >
        Upload
      </button>

      {status && <p className="mt-4">{status}</p>}
    </main>
  );
}