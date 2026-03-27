import React from "react";

export default function StatusPipeline({ stage, message }) {
  return (
    <div
      style={{
        maxWidth: "800px",
        margin: "0 auto",
        padding: "0 20px",
        display: "flex",
        alignItems: "center",
        marginBottom: "16px",
      }}
    >
      <div
        className="animate-fade-in"
        style={{
          display: "inline-flex",
          alignItems: "center",
          padding: "6px 12px",
          backgroundColor: "var(--status-pipeline-bg)",
          border: "1px solid var(--border-light)",
          borderRadius: "20px",
          boxShadow: "0 2px 5px rgba(0,0,0,0.02)",
        }}
      >
        <div
          className="animate-pulse-slow"
          style={{
            width: "12px",
            height: "12px",
            borderRadius: "50%",
            background: "linear-gradient(135deg, #0071e3, #33a1ff)",
            marginRight: "12px",
            boxShadow: "0 0 8px rgba(0,113,227,0.4)",
          }}
        />
        <span
          style={{
            fontSize: "13px",
            fontWeight: 500,
            color: "var(--foreground)",
            letterSpacing: "-0.2px",
          }}
        >
          {message}
        </span>
      </div>
    </div>
  );
}
