import { useRef, useEffect } from 'react';
import { Calendar, GraduationCap, CreditCard, Briefcase, ArrowUpRight } from 'lucide-react';
import { useChat } from '../context/ChatContext';
import MessageBubble from './MessageBubble';
import './ChatWindow.css';

const SUGGESTIONS = [
  {
    icon: Calendar,
    title: 'Academic Calendar',
    desc: 'What is the academic calendar for 2025-26?',
  },
  {
    icon: GraduationCap,
    title: 'Semester Examinations',
    desc: 'When are the upcoming semester examinations?',
  },
  {
    icon: CreditCard,
    title: 'Fee Structure',
    desc: 'What is the fee structure and payment timeline?',
  },
  {
    icon: Briefcase,
    title: 'Placement Statistics',
    desc: 'Tell me about recent placement statistics and recruiters',
  },
];

export default function ChatWindow() {
  const { activeConversation, loading } = useChat();
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeConversation?.messages?.length, loading]);

  const hasMessages = Boolean(activeConversation?.messages?.length);

  return (
    <div className="chat-window">
      <div className="chat-content-container">
        {!hasMessages ? (
          <div className="empty-state-wrapper">
            <div className="empty-state">
              <div className="empty-brand-badge">
                <span className="empty-brand-mark">VAIT</span>
              </div>
              <h2 className="empty-title">VVIT's Artificial Intelligence Technology</h2>
              <p className="empty-hint">
                Your institutional AI assistant for academic regulations, examination schedules, administration policies, and placement records.
              </p>

              <div className="empty-suggestions">
                <span className="suggestion-label">Try asking:</span>
                <div className="suggestion-grid">
                  {SUGGESTIONS.map((item, i) => (
                    <SuggestionCard
                      key={i}
                      icon={item.icon}
                      title={item.title}
                      query={item.desc}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="chat-messages" role="log" aria-live="polite">
            {activeConversation.messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {loading && (
              <div className="message-row message-row-assistant">
                <div className="message-bubble bubble-assistant loading-bubble" aria-label="VAIT is generating a response">
                  <div className="loading-content">
                    <span className="dot-typing">
                      <span />
                      <span />
                      <span />
                    </span>
                    <span className="loading-text">Synthesizing institutional answer...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>
    </div>
  );
}

function SuggestionCard({ icon: Icon, title, query }) {
  const { sendMessage } = useChat();

  return (
    <button
      className="suggestion-card"
      onClick={() => sendMessage(query)}
      title={`Ask: "${query}"`}
      type="button"
    >
      <div className="suggestion-card-header">
        <div className="suggestion-icon-wrap">
          <Icon size={15} />
        </div>
        <span className="suggestion-card-title">{title}</span>
        <ArrowUpRight size={13} className="suggestion-arrow" />
      </div>
      <p className="suggestion-card-desc">{query}</p>
    </button>
  );
}
