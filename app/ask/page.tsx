"use client";

import { useState } from "react";
import ChatMessage from "../../components/ChatMessage";
import Loader from "../../components/Loader";

export default function AskPage() {
  const [messages, setMessages] = useState<any[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);

  const ask = async () => {
    if (!question.trim()) return;
    setLoading(true);

    const userMsg = { role: "user", content: question };
    setMessages((m) => [...m, userMsg]);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const data = await res.json();
      const botMsg = { role: "assistant", content: data.answer || "No answer returned." };
      setMessages((m) => [...m, botMsg]);
    } catch (err) {
      console.error("Frontend fetch error:", err);
      const botMsg = { role: "assistant", content: "Error contacting backend." };
      setMessages((m) => [...m, botMsg]);
    }

    setQuestion("");
    setLoading(false);
  };

  return (
    <main className="max-w-3xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Ask a Question</h1>

      <a href="/ingest" className="text-blue-600 underline">
        Upload Manuals
      </a>

      <div className="space-y-4 mb-6">
        {messages.map((msg, idx) => (
          <ChatMessage key={idx} role={msg.role} content={msg.content} />
        ))}
        {loading && <Loader />}
      </div>

      <div className="flex gap-2">
        <input
          className="flex-1 border rounded px-3 py-2"
          placeholder="e.g. When is my next oil change?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button
          onClick={ask}
          className="bg-blue-600 text-white px-4 py-2 rounded"
        >
          Ask
        </button>
      </div>
    </main>
  );
}