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
        className="animate-fade-in floating-pill"
        style={{
          position: "relative",
          display: "inline-flex",
          alignItems: "center",
          padding: "8px 16px",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: "linear-gradient(90deg, transparent, rgba(0, 113, 227, 0.08), transparent)",
            backgroundSize: "200% 100%",
            animation: "shine 2s infinite linear",
            zIndex: 0,
          }}
        />
        <div
          className="animate-pulse-slow"
          style={{
            width: "14px",
            height: "14px",
            borderRadius: "50%",
            background: "linear-gradient(135deg, #0071e3, #66b2ff)",
            marginRight: "10px",
            boxShadow: "0 0 12px rgba(0, 113, 227, 0.5)",
            zIndex: 1,
          }}
        />
        <span
          style={{
            fontSize: "13.5px",
            fontWeight: 500,
            color: "var(--foreground)",
            letterSpacing: "-0.2px",
            zIndex: 1,
            textShadow: "0 1px 2px rgba(255,255,255,0.8)",
          }}
        >
          {message}
        </span>
      </div>
    </div>
  );
}
