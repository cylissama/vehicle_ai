"use client";

import { useState, useRef, useEffect } from "react";
import ChatMessage from "../../components/ChatMessage";
import Loader from "../../components/Loader";

export default function AskPage() {
  const [messages, setMessages] = useState<any[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const ask = async () => {
    if (!question.trim()) return;
    setLoading(true);

    const userMsg = { role: "user", content: question };
    setMessages((m) => [...m, userMsg]);
    setQuestion("");

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

    setLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      ask();
    }
  };

  return (
    <main className="page-container">
      
      <div className="header-row">
        <h1>Ask a Question</h1>
        <a href="/ingest" className="upload-link">
          Upload Manuals
        </a>
      </div>

      <div className="chat-window">
        {messages.length === 0 && (
          <p className="empty-state">
            No messages yet. Ask something about your vehicle!
          </p>
        )}
        
        {messages.map((msg, idx) => (
          <ChatMessage key={idx} role={msg.role} content={msg.content} />
        ))}
        
        {loading && (
           <div className="message-row assistant">
             <div className="bubble assistant">
               <Loader />
             </div>
           </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <input
          className="chat-input"
          placeholder="e.g. When is my next oil change?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          onClick={ask}
          disabled={loading}
          className="send-button"
        >
          {loading ? "..." : "Ask"}
        </button>
      </div>
    </main>
  );
}