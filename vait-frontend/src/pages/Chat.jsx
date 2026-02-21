import { useState } from 'react';
import Sidebar from '../components/Sidebar';
import TopBar from '../components/TopBar';
import ChatWindow from '../components/ChatWindow';
import ChatInput from '../components/ChatInput';
import './Chat.css';

export default function Chat() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="chat-layout">
      <Sidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen((v) => !v)} />
      <main className="chat-main">
        <TopBar />
        <ChatWindow />
        <ChatInput />
      </main>
    </div>
  );
}
