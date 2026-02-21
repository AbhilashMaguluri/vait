import { useState, useRef } from 'react';
import { useChat } from '../context/ChatContext';
import './ChatInput.css';

const MAX_CHARS = 1000;

export default function ChatInput() {
  const [text, setText] = useState('');
  const { sendMessage, loading } = useChat();
  const inputRef = useRef(null);

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed || loading || trimmed.length > MAX_CHARS) return;
    sendMessage(trimmed);
    setText('');
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const charCount = text.length;
  const isOverLimit = charCount > MAX_CHARS;

  return (
    <div className="chat-input-container">
      <div className="chat-input-wrapper">
        <textarea
          ref={inputRef}
          className="chat-input-field"
          placeholder="Ask VAIT a question..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={loading}
          aria-label="Message input"
        />
        <button
          className="chat-send-btn"
          onClick={handleSend}
          disabled={!text.trim() || loading || isOverLimit}
          aria-label="Send message"
        >
          {loading ? (
            <span className="send-loading" />
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          )}
        </button>
      </div>
      <div className="chat-input-footer">
        <span className={`char-counter ${isOverLimit ? 'char-over' : ''}`}>
          {charCount}/{MAX_CHARS}
        </span>
        {loading && <span className="typing-indicator">VAIT is processing...</span>}
      </div>
    </div>
  );
}
