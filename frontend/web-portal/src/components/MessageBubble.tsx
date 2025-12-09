import React from "react";

interface Props {
  role: "user" | "assistant";
  message: string;
  forms?: string[];
  sources?: string[];
}

const bubbleStyle = (role: "user" | "assistant") => ({
  alignSelf: role === "user" ? "flex-end" : "flex-start",
  background: role === "user" ? "#2563eb" : "#f1f5f9",
  color: role === "user" ? "white" : "#0f172a",
  padding: "12px",
  borderRadius: "12px",
  maxWidth: "75%",
  boxShadow: "0 4px 12px rgba(0,0,0,0.05)",
});

export function MessageBubble({ role, message, forms, sources }: Props) {
  return (
    <div style={bubbleStyle(role)}>
      <div style={{ whiteSpace: "pre-wrap" }}>{message}</div>
      {forms && forms.length > 0 && (
        <div className="forms">Formulaires : {forms.join(", ")}</div>
      )}
      {sources && sources.length > 0 && (
        <div className="sources">Sources : {sources.join(", ")}</div>
      )}
    </div>
  );
}
