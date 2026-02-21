import { useChat } from '../context/ChatContext';
import HistoryList from './HistoryList';
import { CATEGORIES } from '../utils/mockAI';
import './Sidebar.css';

export default function Sidebar({ isOpen, onToggle }) {
  const { createConversation, categoryFilter, setCategoryFilter } = useChat();

  return (
    <>
      <aside className={`sidebar ${isOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
        <div className="sidebar-header">
          <span className="sidebar-brand">VAIT</span>
          <button
            className="sidebar-toggle"
            onClick={onToggle}
            aria-label="Toggle sidebar"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>
        </div>

        <button className="new-chat-btn" onClick={createConversation}>
          New Conversation
        </button>

        <div className="sidebar-filter">
          <label className="filter-label" htmlFor="category-filter">Filter</label>
          <select
            id="category-filter"
            className="filter-select"
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
          >
            <option value="All">All Categories</option>
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        <HistoryList />
      </aside>

      {!isOpen && (
        <button
          className="sidebar-open-btn"
          onClick={onToggle}
          aria-label="Open sidebar"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>
      )}
    </>
  );
}
