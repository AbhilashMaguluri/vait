import { LogOut, PanelLeft, Download, Trash2, User } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useChat } from '../context/ChatContext';
import './TopBar.css';

export default function TopBar({ sidebarOpen, onToggleSidebar }) {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const {
    clearChat,
    exportConversation,
    activeConversation,
  } = useChat();

  const handleLogout = async () => {
    await logout();
    navigate('/login', { replace: true });
  };

  const hasMessages = Boolean(activeConversation?.messages?.length);

  return (
    <header className="topbar" role="banner">
      <div className="topbar-left">
        <button
          className={`topbar-sidebar-toggle ${sidebarOpen ? 'sidebar-is-open' : ''}`}
          onClick={onToggleSidebar}
          title={sidebarOpen ? 'Collapse sidebar' : 'Open sidebar'}
          aria-label={sidebarOpen ? 'Collapse sidebar' : 'Open sidebar'}
        >
          <PanelLeft size={16} />
        </button>

        <div className="topbar-brand-lockup">
          <span className="topbar-title">VAIT</span>
          <span className="topbar-divider">/</span>
          <span className="topbar-subtitle">VVIT's Artificial Intelligence Technology</span>
        </div>
      </div>

      <div className="topbar-center">
        <div className="topbar-badge" title="Official Institutional Mode">
          <span className="topbar-badge-dot" />
          <span className="topbar-badge-text">Official Academic Mode</span>
        </div>
      </div>

      <div className="topbar-right">
        {user && (
          <div className="topbar-user-pill" title={`Logged in as ${user.email}`}>
            <div className="topbar-user-avatar">
              <User size={13} />
            </div>
            <span className="topbar-user-name">
              {user.full_name || user.username}
            </span>
          </div>
        )}

        {activeConversation && hasMessages && (
          <div className="topbar-actions">
            <button
              className="topbar-btn"
              onClick={exportConversation}
              title="Export conversation as JSON"
              aria-label="Export conversation"
            >
              <Download size={13} className="btn-icon" />
              <span className="btn-text">Export</span>
            </button>
            <button
              className="topbar-btn topbar-btn-clear"
              onClick={clearChat}
              title="Clear messages in this conversation"
              aria-label="Clear chat"
            >
              <Trash2 size={13} className="btn-icon" />
              <span className="btn-text">Clear</span>
            </button>
          </div>
        )}

        <button
          className="topbar-icon-btn"
          onClick={handleLogout}
          title="Sign out of VAIT"
          aria-label="Logout"
        >
          <LogOut size={15} />
        </button>
      </div>
    </header>
  );
}
