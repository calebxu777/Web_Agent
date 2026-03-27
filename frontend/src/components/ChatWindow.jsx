"use client";

import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";

export default function ChatWindow({ messages, isTyping, onOpenNickname }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  return (
    <div
      className="custom-scrollbar"
      style={{
        flex: 1,
        overflowY: "auto",
        padding: "24px 0",
        scrollBehavior: "smooth",
      }}
    >
      <div
        style={{
          maxWidth: "800px",
          margin: "0 auto",
          display: "flex",
          flexDirection: "column",
          gap: "32px",
        }}
      >
        {messages.length === 0 && (
          <div
            className="animate-fade-in"
            style={{
              textAlign: "center",
              color: "var(--muted-text)",
              marginTop: "10vh",
              padding: "0 20px",
            }}
          >
            <h2
              style={{
                fontSize: "32px",
                fontWeight: "500",
                color: "var(--foreground)",
                marginBottom: "16px",
                letterSpacing: "-0.5px",
              }}
            >
              How can I help you shop today?
            </h2>
            <p style={{ fontSize: "16px", lineHeight: 1.7 }}>
              Search for products, ask for recommendations, or upload an image to
              find similar items.
            </p>
            <p style={{ fontSize: "14px", marginTop: "12px", color: "var(--muted-text)" }}>
              <button
                onClick={onOpenNickname}
                style={{
                  background: "none",
                  border: "none",
                  color: "var(--primary-accent)",
                  cursor: "pointer",
                  fontFamily: "inherit",
                  fontSize: "14px",
                  fontWeight: 500,
                  padding: 0,
                  textDecoration: "underline",
                  textDecorationColor: "rgba(0, 113, 227, 0.3)",
                  textUnderlineOffset: "3px",
                }}
              >
                Set a nickname
              </button>
              {" "}to save your preferences across sessions.
            </p>
          </div>
        )}

        {messages.map((msg, index) => (
          <MessageBubble key={msg.id || index} message={msg} />
        ))}

        {isTyping && messages.length > 0 && !messages[messages.length - 1]?.content && (
          <div style={{ padding: "0 20px" }}>
            <div className="animate-pulse-slow" style={{ color: "var(--muted-text)", fontSize: "14px" }}>
              Thinking...
            </div>
          </div>
        )}

        <div ref={bottomRef} style={{ height: "2px" }} />
      </div>
    </div>
  );
}

