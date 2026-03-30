"use client";

import { useState, useEffect } from "react";
import ChatWindow from "@/components/ChatWindow";
import ChatInput from "@/components/ChatInput";
import Header from "@/components/Header";
import StatusPipeline from "@/components/StatusPipeline";
import SearchToggle from "@/components/SearchToggle";
import NicknameModal from "@/components/NicknameModal";
import WorksheetPanel from "@/components/WorksheetPanel";

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [pipelineStage, setPipelineStage] = useState(null);
  const [pipelineMsg, setPipelineMsg] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [worksheetState, setWorksheetState] = useState(null);

  // Nickname state
  const [nickname, setNickname] = useState("");
  const [showNicknameModal, setShowNicknameModal] = useState(false);

  // Keep nickname only for the current browser session.
  useEffect(() => {
    localStorage.removeItem("commerce_nickname");
    const saved = sessionStorage.getItem("commerce_nickname");
    if (saved) setNickname(saved);
  }, []);

  const handleNicknameSave = (name, status) => {
    if (name) {
      setNickname(name);
      sessionStorage.setItem("commerce_nickname", name);
    } else {
      // Cleared
      setNickname("");
      sessionStorage.removeItem("commerce_nickname");
      localStorage.removeItem("commerce_nickname");
    }
  };

  const fileToDataUrl = (file) =>
    new Promise((resolve, reject) => {
      if (!file) {
        resolve(null);
        return;
      }
      const reader = new FileReader();
      reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : null);
      reader.onerror = () => reject(reader.error || new Error("Failed to read image file."));
      reader.readAsDataURL(file);
    });

  const sendMessage = async (text, image) => {
    const newMessage = { id: Date.now(), role: "user", content: text };
    if (image) {
      newMessage.imageUrl = URL.createObjectURL(image);
    }

    setMessages((prev) => [...prev, newMessage]);
    setIsTyping(true);
    setPipelineStage("cold_start");
    setPipelineMsg("Loading the model for cold start...");
    setWorksheetState(null);

    try {
      const imageBase64 = image ? await fileToDataUrl(image) : null;
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          hasImage: !!image,
          imageBase64,
          webSearch: webSearchEnabled,
          user_id: nickname || "",  // Send nickname as user_id
        }),
      });

      if (!response.body) return;

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const assistantMsgId = Date.now() + 1;
      let assistantContent = "";
      let assistantProducts = [];
      let hasAddedAssistant = false;
      let sseBuffer = "";

      const upsertAssistant = () => {
        if (!hasAddedAssistant) {
          setMessages((prev) => [
            ...prev,
            { id: assistantMsgId, role: "assistant", content: assistantContent, products: [...assistantProducts] },
          ]);
          hasAddedAssistant = true;
        } else {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, content: assistantContent, products: [...assistantProducts] }
                : m
            )
          );
        }
      };

      const processSseLine = (line) => {
        if (line.trim() === "data: [DONE]") {
          setIsTyping(false);
          setPipelineStage(null);
          return;
        }
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));

            if (data.type === "status") {
              setPipelineStage(data.stage);
              setPipelineMsg(data.message);
            }

            if (data.type === "worksheet_state") {
              setWorksheetState(data.worksheet);
            }

            if (data.type === "products") {
              assistantProducts = data.items;
              upsertAssistant();
            }

            if (data.type === "token") {
              assistantContent += data.content;
              upsertAssistant();
            }
          } catch (err) {
            /* ignore malformed chunks */
          }
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        sseBuffer += decoder.decode(value || new Uint8Array(), { stream: !done });

        const lines = sseBuffer.split("\n");
        sseBuffer = done ? "" : (lines.pop() ?? "");

        for (const line of lines) {
          processSseLine(line);
        }

        if (done) {
          if (sseBuffer.trim()) {
            processSseLine(sseBuffer);
          }
          break;
        }
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsTyping(false);
      setPipelineStage(null);
    }
  };

  return (
    <main
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
      }}
    >
      <Header
        nickname={nickname}
        onUserIconClick={() => setShowNicknameModal(true)}
      />
      <ChatWindow
        messages={messages}
        isTyping={isTyping}
        onOpenNickname={() => setShowNicknameModal(true)}
      />
      {pipelineStage && <StatusPipeline stage={pipelineStage} message={pipelineMsg} />}
      <WorksheetPanel worksheet={worksheetState} />
      <div
        style={{
          position: "sticky",
          bottom: 0,
          padding: "24px 0 32px",
          background: "linear-gradient(to top, var(--background) 50%, transparent)",
          zIndex: 10,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          pointerEvents: "none"
        }}
      >
        <div style={{ width: "100%", maxWidth: "800px", padding: "0 20px", display: "flex", justifyContent: "flex-end", marginBottom: "12px", pointerEvents: "auto" }}>
          <SearchToggle
            webSearchEnabled={webSearchEnabled}
            onToggle={() => setWebSearchEnabled((prev) => !prev)}
            disabled={isTyping}
          />
        </div>
        <div style={{ width: "100%", pointerEvents: "auto" }} className="animate-slide-up">
          <ChatInput onSend={sendMessage} disabled={isTyping} />
        </div>
      </div>

      {/* Nickname Modal */}
      <NicknameModal
        isOpen={showNicknameModal}
        onClose={() => setShowNicknameModal(false)}
        onSave={handleNicknameSave}
        currentNickname={nickname}
      />
    </main>
  );
}
