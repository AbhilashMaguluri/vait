import { useState } from 'react';
import { useChat } from '../context/ChatContext';
import './HistoryList.css';

function formatDate(isoString) {
  const d = new Date(isoString);
  return d.toLocaleDateString('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
}

function formatTime(isoString) {
  const d = new Date(isoString);
  return d.toLocaleTimeString('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  });
}

const CATEGORY_COLORS = {
  Academic: '#2d6a4f',
  Exams: '#8a6d2b',
  Administration: '#4a5899',
  Placements: '#7b3f8d',
};

export default function HistoryList() {
  const {
    groupedConversations,
    activeConversationId,
    setActiveConversationId,
    deleteConversation,
    renameConversation,
  } = useChat();

  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState('');

  const startRename = (conv) => {
    setEditingId(conv.id);
    setEditTitle(conv.title);
  };

  const commitRename = () => {
    if (editingId && editTitle.trim()) {
      renameConversation(editingId, editTitle.trim());
    }
    setEditingId(null);
    setEditTitle('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') commitRename();
    if (e.key === 'Escape') {
      setEditingId(null);
      setEditTitle('');
    }
  };

  const groups = Object.entries(groupedConversations).filter(
    ([, items]) => items.length > 0
  );

  if (groups.length === 0) {
    return (
      <div className="history-empty">
        <p>No conversations yet</p>
      </div>
    );
  }

  return (
    <div className="history-list">
      {groups.map(([label, items]) => (
        <div key={label} className="history-group">
          <h4 className="history-group-label">{label}</h4>
          {items.map((conv) => (
            <div
              key={conv.id}
              className={`history-item ${conv.id === activeConversationId ? 'history-item-active' : ''}`}
              onClick={() => setActiveConversationId(conv.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Enter' && setActiveConversationId(conv.id)}
            >
              <div className="history-item-header">
                {editingId === conv.id ? (
                  <input
                    className="history-rename-input"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onBlur={commitRename}
                    onKeyDown={handleKeyDown}
                    onClick={(e) => e.stopPropagation()}
                    autoFocus
                  />
                ) : (
                  <span className="history-item-title">{conv.title}</span>
                )}
              </div>

              <div className="history-item-meta">
                <span className="history-date">{formatDate(conv.createdAt)}</span>
                <span className="history-time">{formatTime(conv.updatedAt)}</span>
              </div>

              <div className="history-item-details">
                <span
                  className="history-category-tag"
                  style={{
                    color: CATEGORY_COLORS[conv.category] || '#555',
                    borderColor: CATEGORY_COLORS[conv.category] || '#ccc',
                  }}
                >
                  {conv.category}
                </span>
                {conv.confidence && (
                  <span className={`history-confidence history-conf-${conv.confidence.toLowerCase()}`}>
                    {conv.confidence}
                  </span>
                )}
              </div>

              <div className="history-item-actions" onClick={(e) => e.stopPropagation()}>
                <button
                  className="history-action-btn"
                  onClick={() => startRename(conv)}
                  title="Rename"
                  aria-label="Rename conversation"
                >
                  Rename
                </button>
                <button
                  className="history-action-btn history-action-delete"
                  onClick={() => deleteConversation(conv.id)}
                  title="Delete"
                  aria-label="Delete conversation"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
