"use client";

import { useState, useEffect, useRef, useCallback } from "react";

export default function NicknameModal({ isOpen, onClose, onSave, currentNickname }) {
  const [nickname, setNickname] = useState("");
  const [status, setStatus] = useState(null); // null, 'checking', 'available', 'exists', 'error'
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);
  const inputRef = useRef(null);
  const debounceRef = useRef(null);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
    if (isOpen) {
      setNickname(currentNickname || "");
      setStatus(currentNickname ? "exists" : null);
      setMessage(currentNickname ? `Logged in as ${currentNickname}` : "");
    }
  }, [isOpen, currentNickname]);

  const checkNickname = useCallback(async (name) => {
    if (name.length < 2) {
      setStatus(null);
      setMessage("");
      return;
    }
    setStatus("checking");
    try {
      const res = await fetch(`/api/nickname/${encodeURIComponent(name)}/check`);
      const data = await res.json();
      if (data.exists) {
        setStatus("exists");
        setMessage("This nickname exists — you'll be welcomed back!");
      } else {
        setStatus("available");
        setMessage("Available! ✓");
      }
    } catch {
      setStatus("error");
      setMessage("Could not check — backend may be down");
    }
  }, []);

  const handleInputChange = (e) => {
    const val = e.target.value.replace(/[^a-zA-Z0-9_-]/g, ""); // alphanumeric + _ -
    setNickname(val);
    clearTimeout(debounceRef.current);
    if (val.length >= 2) {
      debounceRef.current = setTimeout(() => checkNickname(val), 400);
    } else {
      setStatus(null);
      setMessage("");
    }
  };

  const handleSave = async () => {
    if (nickname.length < 2 || saving) return;
    setSaving(true);
    try {
      const res = await fetch("/api/nickname", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nickname }),
      });
      const data = await res.json();
      if (data.error) {
        setStatus("error");
        setMessage(data.error);
      } else {
        onSave(data.nickname, data.status);
        onClose();
      }
    } catch {
      setStatus("error");
      setMessage("Failed to save — is the backend running?");
    } finally {
      setSaving(false);
    }
  };

  const handleClear = () => {
    onSave("", "cleared");
    onClose();
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && nickname.length >= 2) handleSave();
    if (e.key === "Escape") onClose();
  };

  if (!isOpen) return null;

  const statusColor =
    status === "available" ? "#34c759" :
    status === "exists" ? "#ff9f0a" :
    status === "error" ? "#ff3b30" :
    "var(--muted-text)";

  return (
    <div
      id="nickname-modal-overlay"
      onClick={(e) => { if (e.target.id === "nickname-modal-overlay") onClose(); }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(0, 0, 0, 0.3)",
        backdropFilter: "blur(8px)",
        WebkitBackdropFilter: "blur(8px)",
        animation: "fadeIn 0.2s ease",
      }}
    >
      <div
        className="animate-fade-in-scale"
        style={{
          background: "#ffffff",
          borderRadius: "20px",
          padding: "32px",
          width: "380px",
          maxWidth: "90vw",
          boxShadow: "0 24px 80px rgba(0, 0, 0, 0.15), 0 4px 16px rgba(0, 0, 0, 0.08)",
          border: "1px solid var(--border-light)",
          position: "relative",
        }}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          style={{
            position: "absolute",
            top: "16px",
            right: "16px",
            background: "none",
            border: "none",
            cursor: "pointer",
            fontSize: "18px",
            color: "var(--muted-text)",
            padding: "4px",
            lineHeight: 1,
          }}
        >
          ✕
        </button>

        {/* Icon */}
        <div style={{ textAlign: "center", fontSize: "36px", marginBottom: "8px" }}>
          {currentNickname ? "👋" : "👤"}
        </div>

        {/* Title */}
        <h3 style={{
          textAlign: "center",
          fontSize: "20px",
          fontWeight: 600,
          color: "var(--foreground)",
          marginBottom: "8px",
          letterSpacing: "-0.3px",
        }}>
          {currentNickname ? `Hi, ${currentNickname}!` : "Choose a Nickname"}
        </h3>

        {/* Subtitle */}
        <p style={{
          textAlign: "center",
          fontSize: "14px",
          color: "var(--muted-text)",
          marginBottom: "24px",
          lineHeight: 1.5,
        }}>
          {currentNickname
            ? "Your preferences are being saved across sessions."
            : "Save your shopping preferences across sessions with a nickname."}
        </p>

        {/* Input */}
        <div style={{
          position: "relative",
          marginBottom: "8px",
        }}>
          <input
            ref={inputRef}
            type="text"
            value={nickname}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Enter a nickname..."
            maxLength={30}
            disabled={saving}
            style={{
              width: "100%",
              padding: "14px 16px",
              borderRadius: "12px",
              border: "1.5px solid",
              borderColor: status === "available" ? "#34c75940" :
                          status === "exists" ? "#ff9f0a40" :
                          status === "error" ? "#ff3b3040" :
                          "var(--border-light)",
              fontSize: "15px",
              fontFamily: "inherit",
              outline: "none",
              transition: "all 0.2s ease",
              boxSizing: "border-box",
              background: "#fafafc",
            }}
          />
          {status === "checking" && (
            <div style={{
              position: "absolute",
              right: "14px",
              top: "50%",
              transform: "translateY(-50%)",
              width: "16px",
              height: "16px",
              border: "2px solid var(--border-light)",
              borderTopColor: "var(--primary-accent)",
              borderRadius: "50%",
              animation: "spin 0.6s linear infinite",
            }} />
          )}
        </div>

        {/* Status message */}
        {message && (
          <p style={{
            fontSize: "12.5px",
            color: statusColor,
            marginBottom: "16px",
            marginTop: "4px",
            paddingLeft: "4px",
            transition: "all 0.2s ease",
          }}>
            {message}
          </p>
        )}

        {!message && <div style={{ height: "28px" }} />}

        {/* Buttons */}
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <button
            onClick={handleSave}
            disabled={nickname.length < 2 || saving}
            style={{
              width: "100%",
              padding: "13px",
              borderRadius: "12px",
              border: "none",
              background: nickname.length >= 2
                ? "var(--primary-accent)"
                : "var(--border-light)",
              color: nickname.length >= 2 ? "#fff" : "var(--muted-text)",
              fontSize: "15px",
              fontWeight: 600,
              fontFamily: "inherit",
              cursor: nickname.length >= 2 ? "pointer" : "not-allowed",
              transition: "all 0.2s ease",
              letterSpacing: "-0.2px",
            }}
          >
            {saving ? "Saving..." :
              status === "exists" ? "Continue as " + nickname :
              "Save Nickname"}
          </button>

          {currentNickname && (
            <button
              onClick={handleClear}
              style={{
                width: "100%",
                padding: "10px",
                borderRadius: "12px",
                border: "none",
                background: "transparent",
                color: "#ff3b30",
                fontSize: "13px",
                fontWeight: 500,
                fontFamily: "inherit",
                cursor: "pointer",
              }}
            >
              Clear Nickname
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
