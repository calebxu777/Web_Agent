export default function Header() {
  return (
    <header
      className="glass-panel"
      style={{
        padding: "16px 24px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        position: "sticky",
        top: 0,
        zIndex: 50,
        borderBottom: "1px solid var(--border-light)",
      }}
    >
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
    </header>
  );
}
