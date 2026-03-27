"use client";

export default function SearchToggle({ webSearchEnabled, onToggle, disabled }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "10px",
      }}
    >
      <span
        style={{
          fontSize: "13px",
          fontWeight: 500,
          color: webSearchEnabled ? "var(--muted-text)" : "var(--foreground)",
          transition: "color 0.2s",
        }}
      >
        Local
      </span>

      {/* Toggle Switch */}
      <button
        type="button"
        onClick={onToggle}
        disabled={disabled}
        aria-label="Toggle web search"
        style={{
          position: "relative",
          width: "44px",
          height: "24px",
          borderRadius: "12px",
          border: "none",
          cursor: disabled ? "not-allowed" : "pointer",
          backgroundColor: webSearchEnabled ? "var(--primary-accent)" : "var(--border-light)",
          transition: "background-color 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          padding: 0,
          opacity: disabled ? 0.5 : 1,
        }}
      >
        <div
          style={{
            position: "absolute",
            top: "2px",
            left: webSearchEnabled ? "22px" : "2px",
            width: "20px",
            height: "20px",
            borderRadius: "50%",
            backgroundColor: "#fff",
            boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
            transition: "left 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          }}
        />
      </button>

      <span
        style={{
          fontSize: "13px",
          fontWeight: 500,
          color: webSearchEnabled ? "var(--foreground)" : "var(--muted-text)",
          transition: "color 0.2s",
        }}
      >
        Local + Web
      </span>
    </div>
  );
}
