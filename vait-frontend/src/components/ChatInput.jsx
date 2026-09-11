import { useState, useRef, useEffect } from 'react';
import { ArrowUp } from 'lucide-react';
import { useChat } from '../context/ChatContext';
import './ChatInput.css';

const MAX_CHARS = 1000;

export default function ChatInput() {
  const [text, setText] = useState('');
  const { sendMessage, loading } = useChat();
  const inputRef = useRef(null);

  // Auto-resize textarea based on input length
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  }, [text]);

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed || loading || trimmed.length > MAX_CHARS) return;
    console.log('[VAIT][Debug] User input:', trimmed);
    sendMessage(trimmed);
    setText('');
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const charCount = text.length;
  const isOverLimit = charCount > MAX_CHARS;
  const canSend = Boolean(text.trim()) && !loading && !isOverLimit;

  return (
    <footer className="chat-input-container">
      <div className="chat-input-inner">
        <div className="chat-input-wrapper">
          <textarea
            ref={inputRef}
            className="chat-input-field"
            placeholder="Ask VAIT about courses, exams, fees, or placements..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={loading}
            aria-label="Message input"
          />
          <button
            className={`chat-send-btn ${canSend ? 'send-btn-active' : ''}`}
            onClick={handleSend}
            disabled={!canSend}
            title="Send message (Enter)"
            aria-label="Send message"
            type="button"
          >
            {loading ? (
              <span className="send-loading" aria-label="Processing" />
            ) : (
              <ArrowUp size={16} strokeWidth={2.4} />
            )}
          </button>
        </div>

        <div className="chat-input-footer">
          <span className="input-shortcut-hint">
            <strong>Enter</strong> to send &bull; <strong>Shift+Enter</strong> for new line
          </span>
          <div className="input-status-right">
            {loading && <span className="typing-indicator">VAIT is thinking...</span>}
            <span className={`char-counter ${isOverLimit ? 'char-over' : ''}`}>
              {charCount > 0 && `${charCount}/${MAX_CHARS}`}
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
