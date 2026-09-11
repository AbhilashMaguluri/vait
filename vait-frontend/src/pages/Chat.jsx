import { useState, useEffect, useCallback } from 'react';
import Sidebar from '../components/Sidebar';
import TopBar from '../components/TopBar';
import ChatWindow from '../components/ChatWindow';
import ChatInput from '../components/ChatInput';
import './Chat.css';

export default function Chat() {
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    if (typeof window !== 'undefined') {
      return window.innerWidth > 768;
    }
    return true;
  });

  // Handle window resize to auto-collapse/restore if appropriate
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth <= 768 && sidebarOpen) {
        // On small screens, keep drawer closed by default unless user opened it
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [sidebarOpen]);

  const handleToggle = useCallback(() => {
    setSidebarOpen((v) => !v);
  }, []);

  const handleClose = useCallback(() => {
    setSidebarOpen(false);
  }, []);

  return (
    <div className="chat-layout">
      {/* Mobile Drawer Backdrop */}
      {sidebarOpen && (
        <div
          className="sidebar-backdrop"
          onClick={handleClose}
          aria-hidden="true"
        />
      )}

      <Sidebar
        isOpen={sidebarOpen}
        onToggle={handleToggle}
        onClose={handleClose}
      />

      <main className="chat-main">
        <TopBar
          sidebarOpen={sidebarOpen}
          onToggleSidebar={handleToggle}
        />
        <ChatWindow />
        <ChatInput />
      </main>
    </div>
  );
}
