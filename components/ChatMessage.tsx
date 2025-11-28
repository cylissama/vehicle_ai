import React from "react";
import './ChatMessage.css';

interface ChatMessageProps {
  role: string;
  content: string;
}

export default function ChatMessage({ role, content }: ChatMessageProps) {
  const isUser = role === "user";

  return (
    // The parent div decides alignment (left vs right)
    <div className={`message-row ${isUser ? "user" : "assistant"}`}>
      
      {/* The bubble handles color and shape */}
      <div className={`bubble ${isUser ? "user" : "assistant"}`}>
        
        <span className={`sender-name ${isUser ? "user" : "assistant"}`}>
          {isUser ? "You" : "Assistant"}
        </span>

        <div className="message-content">
          {content}
        </div>

      </div>
    </div>
  );
}