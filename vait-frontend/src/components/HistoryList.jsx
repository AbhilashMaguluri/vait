import { useState } from 'react';
import { Pencil, Trash2, Check, X } from 'lucide-react';
import { useChat } from '../context/ChatContext';
import './HistoryList.css';

function formatDate(isoString) {
  const d = new Date(isoString);
  return d.toLocaleDateString('en-IN', {
    timeZone: 'Asia/Kolkata',
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
}

function formatTime(isoString) {
  const d = new Date(isoString);
  return d.toLocaleTimeString('en-IN', {
    timeZone: 'Asia/Kolkata',
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  });
}

export default function HistoryList({ onSelect }) {
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

  const cancelRename = () => {
    setEditingId(null);
    setEditTitle('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') commitRename();
    if (e.key === 'Escape') cancelRename();
  };

  const handleItemClick = (id) => {
    setActiveConversationId(id);
    onSelect?.();
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
    <div className="history-list" role="navigation" aria-label="Conversation list">
      {groups.map(([label, items]) => (
        <div key={label} className="history-group">
          <h4 className="history-group-label">{label}</h4>
          {items.map((conv) => {
            const isActive = conv.id === activeConversationId;
            const categoryClass = `category-${(conv.category || 'academic').toLowerCase()}`;

            return (
              <div
                key={conv.id}
                className={`history-item ${isActive ? 'history-item-active' : ''}`}
                onClick={() => handleItemClick(conv.id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === 'Enter' && handleItemClick(conv.id)}
                aria-current={isActive ? 'true' : undefined}
              >
                <div className="history-item-main">
                  <div className="history-item-header">
                    {editingId === conv.id ? (
                      <div className="history-rename-wrap" onClick={(e) => e.stopPropagation()}>
                        <input
                          className="history-rename-input"
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          onKeyDown={handleKeyDown}
                          autoFocus
                          aria-label="Edit title"
                        />
                        <button
                          className="history-rename-action-btn"
                          onClick={commitRename}
                          title="Save"
                          aria-label="Save title"
                        >
                          <Check size={12} />
                        </button>
                        <button
                          className="history-rename-action-btn"
                          onClick={cancelRename}
                          title="Cancel"
                          aria-label="Cancel editing"
                        >
                          <X size={12} />
                        </button>
                      </div>
                    ) : (
                      <span className="history-item-title" title={conv.title}>
                        {conv.title}
                      </span>
                    )}
                  </div>

                  <div className="history-item-meta">
                    <span className="history-date">{formatDate(conv.createdAt)}</span>
                    <span className="history-bullet">•</span>
                    <span className="history-time">{formatTime(conv.updatedAt)}</span>
                  </div>

                  <div className="history-item-details">
                    <span className={`history-category-tag ${categoryClass}`}>
                      {conv.category}
                    </span>
                    {conv.confidence && (
                      <span className={`history-confidence history-conf-${conv.confidence.toLowerCase()}`}>
                        {conv.confidence}
                      </span>
                    )}
                  </div>
                </div>

                {editingId !== conv.id && (
                  <div className="history-item-actions" onClick={(e) => e.stopPropagation()}>
                    <button
                      className="history-action-btn"
                      onClick={() => startRename(conv)}
                      title="Rename conversation"
                      aria-label="Rename conversation"
                    >
                      <Pencil size={11} />
                    </button>
                    <button
                      className="history-action-btn history-action-delete"
                      onClick={() => deleteConversation(conv.id)}
                      title="Delete conversation"
                      aria-label="Delete conversation"
                    >
                      <Trash2 size={11} />
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
