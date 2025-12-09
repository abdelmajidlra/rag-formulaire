import React from "react";

import { ChatWindow } from "../components/ChatWindow";

export default function App() {
  return (
    <div className="app-shell">
      <div className="header">
        <h1>Assistant formulaires IRCC</h1>
        <div style={{ color: "#475569" }}>Mode démo (auth AAD à venir)</div>
      </div>
      <ChatWindow />
    </div>
  );
}
