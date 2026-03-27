"use client";

export default function Header({ nickname, onUserIconClick }) {
  const hasNickname = !!nickname;

  return (
    <header
      className="glass-panel"
      style={{
        padding: "16px 24px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "sticky",
        top: 0,
        zIndex: 50,
        borderBottom: "1px solid var(--border-light)",
      }}
    >
      {/* Spacer for centering */}
      <div style={{ width: "40px" }} />

      {/* Title */}
      <h1
        style={{
          fontSize: "18px",
          fontWeight: 600,
          letterSpacing: "-0.3px",
          color: "var(--foreground)",
        }}
      >
        Compound AI Commerce
      </h1>

      {/* User icon */}
      <button
        id="user-icon-button"
        onClick={onUserIconClick}
        title={hasNickname ? `Logged in as ${nickname}` : "Set a nickname"}
        style={{
          width: "36px",
          height: "36px",
          borderRadius: "50%",
          border: hasNickname ? "2px solid var(--primary-accent)" : "1.5px solid var(--border-light)",
          background: hasNickname
            ? "linear-gradient(135deg, #0071e3, #5ac8fa)"
            : "var(--background)",
          color: hasNickname ? "#fff" : "var(--muted-text)",
          fontSize: hasNickname ? "15px" : "16px",
          fontWeight: 600,
          fontFamily: "inherit",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: "pointer",
          transition: "all 0.25s cubic-bezier(0.25, 1, 0.5, 1)",
          position: "relative",
          lineHeight: 1,
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = "scale(1.08)";
          if (!hasNickname) {
            e.currentTarget.style.borderColor = "var(--primary-accent)";
            e.currentTarget.style.color = "var(--primary-accent)";
          }
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = "scale(1)";
          if (!hasNickname) {
            e.currentTarget.style.borderColor = "var(--border-light)";
            e.currentTarget.style.color = "var(--muted-text)";
          }
        }}
      >
        {hasNickname
          ? nickname.charAt(0).toUpperCase()
          : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
          )
        }
      </button>
    </header>
  );
}
