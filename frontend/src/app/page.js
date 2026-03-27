"use client";

import { useState } from "react";
import ChatWindow from "@/components/ChatWindow";
import ChatInput from "@/components/ChatInput";
import Header from "@/components/Header";
import StatusPipeline from "@/components/StatusPipeline";
import SearchToggle from "@/components/SearchToggle";

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [pipelineStage, setPipelineStage] = useState(null);
  const [pipelineMsg, setPipelineMsg] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);

  const sendMessage = async (text, image) => {
    const newMessage = { id: Date.now(), role: "user", content: text };
    if (image) {
      newMessage.imageUrl = URL.createObjectURL(image);
    }

    setMessages((prev) => [...prev, newMessage]);
    setIsTyping(true);
    setPipelineStage("cold_start");
    setPipelineMsg("Loading the model for cold start...");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          hasImage: !!image,
          webSearch: webSearchEnabled,
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
        backgroundColor: "var(--background)",
      }}
    >
      <Header />
      <ChatWindow messages={messages} isTyping={isTyping} />
      {pipelineStage && <StatusPipeline stage={pipelineStage} message={pipelineMsg} />}
      <div
        className="glass-panel"
        style={{ position: "sticky", bottom: 0, padding: "16px 0 24px", zIndex: 10 }}
      >
        <div style={{ maxWidth: "800px", margin: "0 auto", padding: "0 20px 12px", display: "flex", justifyContent: "flex-end" }}>
          <SearchToggle
            webSearchEnabled={webSearchEnabled}
            onToggle={() => setWebSearchEnabled((prev) => !prev)}
            disabled={isTyping}
          />
        </div>
        <ChatInput onSend={sendMessage} disabled={isTyping} />
      </div>
    </main>
  );
}
