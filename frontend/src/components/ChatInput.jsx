"use client";

import { useState, useRef } from "react";

const SUPPORTED_IMAGE_TYPES = new Set(["image/png", "image/jpeg"]);
const SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg"];
const UNSUPPORTED_IMAGE_MESSAGE =
  "That image format isn't supported yet. Please upload PNG, JPG, or JPEG.";

function isSupportedImageFile(file) {
  if (!file) return false;
  if (file.type && SUPPORTED_IMAGE_TYPES.has(file.type.toLowerCase())) {
    return true;
  }
  const name = String(file.name || "").toLowerCase();
  return SUPPORTED_EXTENSIONS.some((extension) => name.endsWith(extension));
}

export default function ChatInput({ onSend, onUnsupportedImage, disabled }) {
  const [text, setText] = useState("");
  const [image, setImage] = useState(null);
  const fileInputRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if ((!text.trim() && !image) || disabled) return;
    onSend(text, image);
    setText("");
    setImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const nextImage = e.target.files[0];
      if (!isSupportedImageFile(nextImage)) {
        setImage(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
        e.target.value = "";
        onUnsupportedImage?.(UNSUPPORTED_IMAGE_MESSAGE);
        return;
      }
      setImage(nextImage);
      e.target.value = "";
    }
  };

  const previewUrl = image ? URL.createObjectURL(image) : null;

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "0 20px" }}>
      {image && (
        <div
          className="animate-fade-in"
          style={{
            position: "relative",
            display: "inline-block",
            marginBottom: "12px",
            padding: "8px",
            border: "1px solid var(--border-light)",
            borderRadius: "12px",
            backgroundColor: "#fff",
          }}
        >
          <span
            onClick={() => {
              setImage(null);
              if (fileInputRef.current) {
                fileInputRef.current.value = "";
              }
            }}
            style={{
              position: "absolute",
              top: "-6px",
              right: "-6px",
              background: "var(--foreground)",
              color: "#fff",
              borderRadius: "50%",
              width: "20px",
              height: "20px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "12px",
              cursor: "pointer",
              zIndex: 10,
            }}
          >
            ✕
          </span>
          <img
            src={previewUrl}
            alt="Preview"
            style={{ height: "60px", borderRadius: "6px" }}
          />
        </div>
      )}

      <form
        className="floating-pill"
        onSubmit={handleSubmit}
        style={{
          display: "flex",
          alignItems: "flex-end",
          padding: "6px 6px 6px 12px",
        }}
      >
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            padding: "8px",
            color: "var(--muted-text)",
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
          title="Upload image"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21 15 16 10 5 21" />
          </svg>
        </button>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".png,.jpg,.jpeg,image/png,image/jpeg"
          style={{ display: "none" }}
        />

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything..."
          disabled={disabled}
          style={{
            flex: 1,
            border: "none",
            outline: "none",
            backgroundColor: "transparent",
            resize: "none",
            padding: "10px 8px",
            fontSize: "16px",
            lineHeight: "1.4",
            fontFamily: "inherit",
            maxHeight: "150px",
            color: "var(--foreground)",
          }}
          rows={1}
        />

        <button
          type="submit"
          disabled={(!text.trim() && !image) || disabled}
          style={{
            background: !text.trim() && !image ? "var(--border-light)" : "var(--foreground)",
            color: "var(--background)",
            border: "none",
            borderRadius: "50%",
            width: "36px",
            height: "36px",
            cursor: (!text.trim() && !image) || disabled ? "not-allowed" : "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            transition: "background-color 0.2s ease",
            marginLeft: "8px",
            marginBottom: "2px",
          }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </form>
      <div style={{ textAlign: "center", marginTop: "10px", fontSize: "11px", color: "var(--muted-text)" }}>
        AI can make mistakes. Verify important information.
      </div>
    </div>
  );
}
