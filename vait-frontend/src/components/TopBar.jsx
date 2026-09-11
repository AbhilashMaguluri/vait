import { LogOut } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useChat } from '../context/ChatContext';
import './TopBar.css';

export default function TopBar() {
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

  return (
    <header className="topbar">
      <div className="topbar-left">
        <h1 className="topbar-title">VAIT</h1>
        <span className="topbar-subtitle">VVIT's Artificial Intelligence Technology</span>
      </div>

      <div className="topbar-center">
        <span className="topbar-badge">Official Academic Mode</span>
      </div>

      <div className="topbar-right">
        {user && (
          <span className="topbar-user" title={user.email}>
            {user.full_name || user.username}
          </span>
        )}
        {activeConversation && (
          <>
            <button className="topbar-btn" onClick={exportConversation} title="Export conversation">
              Export
            </button>
            <button className="topbar-btn topbar-btn-clear" onClick={clearChat} title="Clear chat">
              Clear
            </button>
          </>
        )}
        <button className="topbar-icon-btn" onClick={handleLogout} title="Logout" aria-label="Logout">
          <LogOut size={15} />
        </button>
      </div>
    </header>
  );
}
