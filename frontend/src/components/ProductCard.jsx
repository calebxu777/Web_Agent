"use client";

import React, { useState } from "react";

function ThumbButton({ type, active, onClick }) {
  const isUp = type === "up";
  const emoji = isUp ? "👍" : "👎";
  const activeColor = isUp ? "#34c759" : "#ff3b30";

  return (
    <button
      id={`thumb-${type}`}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        onClick();
      }}
      title={isUp ? "Add to catalog" : "Not relevant"}
      style={{
        background: active ? activeColor + "18" : "transparent",
        border: active ? `1px solid ${activeColor}40` : "1px solid transparent",
        borderRadius: "8px",
        padding: "3px 6px",
        cursor: "pointer",
        fontSize: "13px",
        transition: "all 0.2s ease",
        opacity: active ? 1 : 0.5,
        transform: active ? "scale(1.1)" : "scale(1)",
        lineHeight: 1,
      }}
      onMouseEnter={(e) => {
        if (!active) e.currentTarget.style.opacity = "0.85";
      }}
      onMouseLeave={(e) => {
        if (!active) e.currentTarget.style.opacity = "0.5";
      }}
    >
      {emoji}
    </button>
  );
}

export default function ProductCard({ product }) {
  const { title, price, description, image, source, url } = product;
  const [vote, setVote] = useState(null); // null | "up" | "down"

  const handleVote = async (newVote) => {
    // Toggle off if already selected
    const finalVote = vote === newVote ? null : newVote;
    setVote(finalVote);

    if (!finalVote) return; // Toggled off, no API call

    // Fire-and-forget — don't block the UI
    try {
      fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ product, vote: finalVote }),
      });
    } catch {
      // Silent failure — don't disrupt UX
    }
  };

  const cardContent = (
    <div
      className="animate-fade-in"
      style={{
        display: "flex",
        gap: "14px",
        padding: "14px",
        borderRadius: "16px",
        border: "1px solid var(--border-light)",
        backgroundColor: "#fff",
        transition: "border-color 0.2s, box-shadow 0.2s",
        cursor: url ? "pointer" : "default",
        maxWidth: "440px",
        width: "100%",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = "var(--border-hover)";
        e.currentTarget.style.boxShadow = "0 4px 12px rgba(0,0,0,0.06)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = "var(--border-light)";
        e.currentTarget.style.boxShadow = "none";
      }}
    >
      {/* Product Image */}
      <div
        style={{
          width: "88px",
          height: "88px",
          minWidth: "88px",
          borderRadius: "12px",
          overflow: "hidden",
          backgroundColor: "var(--bubble-user-bg)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {image ? (
          <img
            src={image}
            alt={title}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
            }}
            onError={(e) => {
              e.target.style.display = "none";
              e.target.parentNode.innerHTML = `
                <div style="display:flex;align-items:center;justify-content:center;width:100%;height:100%;color:var(--muted-text);font-size:24px;">
                  📦
                </div>
              `;
            }}
          />
        ) : (
          <span style={{ fontSize: "28px" }}>📦</span>
        )}
      </div>

      {/* Product Info */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          gap: "4px",
          flex: 1,
          minWidth: 0,
        }}
      >
        {/* Title + Source Badge */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "6px",
          }}
        >
          <span
            style={{
              fontSize: "14.5px",
              fontWeight: 600,
              color: "var(--foreground)",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {title}
          </span>
          {source === "web" && (
            <span
              title="Found on the web"
              style={{
                fontSize: "11px",
                padding: "1px 6px",
                borderRadius: "6px",
                backgroundColor: "#e8f4fd",
                color: "#0071e3",
                fontWeight: 500,
                whiteSpace: "nowrap",
              }}
            >
              Web
            </span>
          )}
        </div>

        {/* Price */}
        {price != null && (
          <span
            style={{
              fontSize: "15px",
              fontWeight: 600,
              color: "var(--foreground)",
              letterSpacing: "-0.2px",
            }}
          >
            ${typeof price === "number" ? price.toFixed(2) : price}
          </span>
        )}

        {/* Description */}
        {description && (
          <p
            style={{
              fontSize: "12.5px",
              color: "var(--muted-text)",
              lineHeight: 1.4,
              margin: 0,
              display: "-webkit-box",
              WebkitLineClamp: 2,
              WebkitBoxOrient: "vertical",
              overflow: "hidden",
            }}
          >
            {description}
          </p>
        )}

        {/* Feedback Buttons */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "4px",
            marginTop: "2px",
          }}
        >
          <ThumbButton
            type="up"
            active={vote === "up"}
            onClick={() => handleVote("up")}
          />
          <ThumbButton
            type="down"
            active={vote === "down"}
            onClick={() => handleVote("down")}
          />
          {vote === "up" && source === "web" && (
            <span
              style={{
                fontSize: "10.5px",
                color: "#34c759",
                fontWeight: 500,
                marginLeft: "4px",
                animation: "fadeIn 0.3s ease",
              }}
            >
              Added to catalog
            </span>
          )}
        </div>
      </div>
    </div>
  );

  if (url) {
    return (
      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        style={{ textDecoration: "none", color: "inherit" }}
      >
        {cardContent}
      </a>
    );
  }

  return cardContent;
}
