import React from "react";
import ProductCard from "./ProductCard";

/**
 * Minimal inline markdown renderer for bold, italic, and list items.
 */
function renderInlineMarkdown(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    const italicParts = part.split(/(\*[^*]+\*)/g);
    return italicParts.map((ip, j) => {
      if (ip.startsWith("*") && ip.endsWith("*") && ip.length > 2) {
        return <em key={`${i}-${j}`}>{ip.slice(1, -1)}</em>;
      }
      return <React.Fragment key={`${i}-${j}`}>{ip}</React.Fragment>;
    });
  });
}

function renderLine(line, idx) {
  if (line.startsWith("- ")) {
    return (
      <li key={idx} style={{ marginLeft: "16px", listStyleType: "disc", marginBottom: "4px" }}>
        {renderInlineMarkdown(line.slice(2))}
      </li>
    );
  }
  if (line.trim() === "") {
    return <div key={idx} style={{ height: "8px" }} />;
  }
  return (
    <p key={idx} style={{ margin: 0, marginBottom: "4px" }}>
      {renderInlineMarkdown(line)}
    </p>
  );
}

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <div
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        padding: "0 20px",
      }}
    >
      <div
        className="animate-fade-in"
        style={{
          backgroundColor: isUser ? "var(--bubble-user-bg)" : "var(--bubble-bot-bg)",
          color: isUser ? "var(--bubble-user-fg)" : "var(--bubble-bot-fg)",
          padding: isUser ? "12px 20px" : "0px 0px",
          borderRadius: isUser ? "20px 20px 4px 20px" : "16px",
          maxWidth: "85%",
          fontSize: "15.5px",
          letterSpacing: "-0.1px",
          lineHeight: 1.6,
        }}
      >
        {!isUser && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              marginBottom: "8px",
            }}
          >
            <div
              style={{
                width: "24px",
                height: "24px",
                borderRadius: "50%",
                backgroundColor: "var(--foreground)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "var(--background)",
                fontWeight: "600",
                fontSize: "12px",
              }}
            >
              AI
            </div>
            <span
              style={{
                fontSize: "13px",
                fontWeight: "500",
                color: "var(--muted-text)",
              }}
            >
              Master Brain
            </span>
          </div>
        )}

        {message.imageUrl && (
          <img
            src={message.imageUrl}
            alt="Uploaded"
            style={{
              maxWidth: "250px",
              borderRadius: "12px",
              marginBottom: "12px",
              border: "1px solid var(--border-light)",
              display: "block",
            }}
          />
        )}

        {/* Product Cards */}
        {message.products && message.products.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginBottom: "16px" }}>
            {message.products.map((product, i) => (
              <ProductCard key={product.id || i} product={product} />
            ))}
          </div>
        )}

        {/* Text content */}
        {message.content && (
          <div className="prose">
            {message.content.split("\n").map((line, i) => renderLine(line, i))}
          </div>
        )}
      </div>
    </div>
  );
}
