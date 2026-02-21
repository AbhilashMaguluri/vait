import { useRef, useEffect } from 'react';
import { useChat } from '../context/ChatContext';
import MessageBubble from './MessageBubble';
import './ChatWindow.css';

export default function ChatWindow() {
  const { activeConversation, loading } = useChat();
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeConversation?.messages?.length, loading]);

  if (!activeConversation || activeConversation.messages.length === 0) {
    return (
      <div className="chat-window chat-window-empty">
        <div className="empty-state">
          <h2 className="empty-title">VAIT</h2>
          <p className="empty-subtitle">
            VVIT's Artificial Intelligence Technology
          </p>
          <p className="empty-hint">
            Ask a question about academics, examinations, administration, or placements.
          </p>
          <div className="empty-suggestions">
            <span className="suggestion-label">Try asking:</span>
            <div className="suggestion-items">
              <SuggestionChip text="What is the academic calendar for 2025-26?" />
              <SuggestionChip text="When are the semester examinations?" />
              <SuggestionChip text="What is the fee structure?" />
              <SuggestionChip text="Tell me about placement statistics" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-window">
      <div className="chat-messages">
        {activeConversation.messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {loading && (
          <div className="message-row message-row-assistant">
            <div className="message-bubble bubble-assistant loading-bubble">
              <span className="dot-typing">
                <span />
                <span />
                <span />
              </span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function SuggestionChip({ text }) {
  const { sendMessage } = useChat();

  return (
    <button className="suggestion-chip" onClick={() => sendMessage(text)}>
      {text}
    </button>
  );
}
