import React, { useMemo, useState } from "react";

import { queryIrccApi, QueryResponse } from "../services/apiClient";
import { MessageBubble } from "./MessageBubble";
import { Loader } from "./Loader";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  forms?: string[];
  sources?: string[];
}

export function ChatWindow() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!input.trim()) return;
    setError(null);
    const question = input.trim();
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    try {
      const result: QueryResponse = await queryIrccApi({ question });
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: result.answer,
          forms: result.forms,
          sources: result.sources,
        },
      ]);
    } catch (err) {
      console.error(err);
      setError("Erreur lors de l'appel API");
    } finally {
      setLoading(false);
    }
  };

  const chatHistory = useMemo(
    () =>
      messages.map((msg, idx) => (
        <MessageBubble
          key={idx}
          role={msg.role}
          message={msg.content}
          forms={msg.forms}
          sources={msg.sources}
        />
      )),
    [messages]
  );

  return (
    <div className="chat-container">
      <div className="message-list">{chatHistory}</div>
      {loading && <Loader />}
      {error && <div style={{ color: "red" }}>{error}</div>}
      <input
        type="text"
        placeholder="Posez votre question sur un formulaire IRCC"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
      />
      <button onClick={handleSubmit} disabled={loading}>
        Envoyer
      </button>
    </div>
  );
}
