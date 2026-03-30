"use client";

import React from "react";

function prettyLabel(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export default function WorksheetPanel({ worksheet }) {
  if (!worksheet) return null;

  const entries = Object.entries(worksheet.values || {});
  const missing = worksheet.missing_required_fields || [];
  const resultCounts = worksheet.result_counts || {};
  const rerankedCount = Number(resultCounts.reranked_count || 0);

  return (
    <div
      className="worksheet-panel animate-fade-in"
      style={{
        maxWidth: "800px",
        margin: "0 auto 16px",
        padding: "0 20px",
      }}
    >
      <div
        style={{
          background: "rgba(255, 255, 255, 0.78)",
          border: "1px solid rgba(0, 113, 227, 0.10)",
          borderRadius: "18px",
          boxShadow: "0 8px 28px rgba(9, 30, 66, 0.06)",
          backdropFilter: "blur(16px)",
          padding: "14px 16px",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "12px",
            marginBottom: entries.length || missing.length ? "12px" : 0,
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span
              style={{
                fontSize: "11px",
                fontWeight: 700,
                letterSpacing: "0.08em",
                color: "var(--primary-accent)",
                textTransform: "uppercase",
              }}
            >
              Worksheet
            </span>
            <span
              style={{
                fontSize: "15px",
                fontWeight: 600,
                color: "var(--foreground)",
              }}
            >
              {prettyLabel(worksheet.name)}
            </span>
          </div>

          <span
            style={{
              fontSize: "12px",
              color: missing.length ? "#b26a00" : "#227a44",
              background: missing.length ? "rgba(255, 159, 10, 0.12)" : "rgba(52, 199, 89, 0.12)",
              borderRadius: "999px",
              padding: "5px 10px",
              fontWeight: 600,
            }}
          >
            {prettyLabel(worksheet.status)}
          </span>
        </div>

        {entries.length > 0 && (
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "8px",
              marginBottom: missing.length || rerankedCount >= 2 ? "12px" : 0,
            }}
          >
            {entries.map(([key, value]) => (
              <span
                key={key}
                style={{
                  fontSize: "12px",
                  color: "var(--foreground)",
                  background: "rgba(0, 0, 0, 0.04)",
                  border: "1px solid rgba(0, 0, 0, 0.05)",
                  borderRadius: "999px",
                  padding: "6px 10px",
                }}
              >
                <strong style={{ fontWeight: 600 }}>{prettyLabel(key)}:</strong>{" "}
                {Array.isArray(value) ? value.join(", ") : String(value)}
              </span>
            ))}
          </div>
        )}

        {missing.length > 0 && (
          <p
            style={{
              margin: 0,
              fontSize: "12.5px",
              color: "#8a5a00",
              lineHeight: 1.5,
            }}
          >
            Missing: {missing.map(prettyLabel).join(", ")}
          </p>
        )}

        {worksheet.name === "product_search" && rerankedCount >= 2 && (
          <p
            style={{
              margin: missing.length ? "10px 0 0" : 0,
              fontSize: "12.5px",
              color: "var(--muted-text)",
              lineHeight: 1.5,
            }}
          >
            You can now ask for a comparison, like <strong>compare the first two</strong>.
          </p>
        )}
      </div>
    </div>
  );
}
