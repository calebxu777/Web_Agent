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

  // Load nickname from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("commerce_nickname");
    if (saved) setNickname(saved);
  }, []);

  const handleNicknameSave = (name, status) => {
    if (name) {
      setNickname(name);
      localStorage.setItem("commerce_nickname", name);
    } else {
      // Cleared
      setNickname("");
      localStorage.removeItem("commerce_nickname");
    }
  };

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
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          hasImage: !!image,
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

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.trim() === "data: [DONE]") {
            setIsTyping(false);
            setPipelineStage(null);
            break;
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
