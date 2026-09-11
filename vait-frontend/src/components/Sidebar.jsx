import { Plus, PanelLeftClose, Filter } from 'lucide-react';
import { useChat } from '../context/ChatContext';
import HistoryList from './HistoryList';
import { CATEGORIES } from '../utils/mockAI';
import './Sidebar.css';

export default function Sidebar({ isOpen, onToggle, onClose }) {
  const { createConversation, categoryFilter, setCategoryFilter } = useChat();

  const handleNewConversation = () => {
    createConversation();
    if (typeof window !== 'undefined' && window.innerWidth <= 768) {
      onClose?.();
    }
  };

  const handleSelectConversation = () => {
    if (typeof window !== 'undefined' && window.innerWidth <= 768) {
      onClose?.();
    }
  };

  return (
    <aside
      className={`sidebar ${isOpen ? 'sidebar-open' : 'sidebar-closed'}`}
      aria-label="Conversation history"
    >
      <div className="sidebar-header">
        <div className="sidebar-brand-group">
          <span className="sidebar-brand">VAIT</span>
          <span className="sidebar-version-pill">Academic</span>
        </div>
        <button
          className="sidebar-toggle"
          onClick={onToggle}
          title="Collapse sidebar"
          aria-label="Collapse sidebar"
        >
          <PanelLeftClose size={16} />
        </button>
      </div>

      <div className="sidebar-action-area">
        <button
          className="new-chat-btn"
          onClick={handleNewConversation}
          title="Start a new conversation"
        >
          <Plus size={15} className="new-chat-icon" />
          <span>New Conversation</span>
        </button>
      </div>

      <div className="sidebar-filter">
        <div className="filter-label-group">
          <Filter size={11} className="filter-icon" />
          <label className="filter-label" htmlFor="category-filter">
            Filter
          </label>
        </div>
        <select
          id="category-filter"
          className="filter-select"
          value={categoryFilter}
          onChange={(e) => setCategoryFilter(e.target.value)}
          aria-label="Filter conversations by category"
        >
          <option value="All">All Categories</option>
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>

      <HistoryList onSelect={handleSelectConversation} />
    </aside>
  );
}
