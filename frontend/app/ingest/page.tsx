"use client";

import { useState } from "react";

export default function IngestPage() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>("");
  const [url, setUrl] = useState("");
  const [statusURL, setStatusURL] = useState("");

  const upload = async () => {
    if (!file) {
      setStatus("Please select a file.");
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

      setStatus(`Upload successful! Ingested ${data.chunks_created || 0} chunks.`);
    } catch (err) {
      console.error(err);
      setStatus("Upload failed. Server unreachable.");
    }
  };

  const ingestUrl = async () => {
    if (!url) {
      setStatusURL("Please enter a URL.");
      return;
    }

    setStatusURL("Processing URL...");

    try {
      const res = await fetch("/api/ingest_url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      const data = await res.json();

      if (!res.ok) {
        setStatusURL("Error: " + data.error);
        return;
      }

      setStatusURL(`Success! Ingested ${data.chunks || 0} chunks.`);
    } catch (err) {
      console.error(err);
      setStatusURL("Failed. Server unreachable.");
    }
  };

  return (
    <main className="max-w-lg mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">Upload a Manual</h1>

      <input
        type="file"
        accept=".pdf,.doc,.docx,.txt,.rtf,.md,.html"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
        className="mb-4 block"
      />

      <button
        onClick={upload}
        className="bg-blue-600 text-white px-4 py-2 rounded"
      >
        Upload File
      </button>

      {status && <p className="mt-4">{status}</p>}

      <hr className="my-6" />

      <h1 className="text-2xl font-bold mb-4">Ingest a URL</h1>

      <input
        type="text"
        value={url}
        placeholder="https://example.com/article"
        onChange={(e) => setUrl(e.target.value)}
        className="border p-2 w-full mb-4"
      />

      <button
        onClick={ingestUrl}
        className="bg-green-600 text-white px-4 py-2 rounded"
      >
        Ingest URL
      </button>

      {statusURL && <p className="mt-4">{statusURL}</p>}
    </main>
  );
}