import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { FileText, ShieldCheck, Clock } from 'lucide-react';
import './MessageBubble.css';

function formatTime(isoString) {
  if (!isoString) return '';
  const d = new Date(isoString);
  return d.toLocaleTimeString('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  });
}

function ConfidenceBadge({ level }) {
  const classMap = {
    High: 'confidence-high',
    Medium: 'confidence-medium',
    Low: 'confidence-low',
  };

  return (
    <span className={`confidence-badge ${classMap[level] || ''}`}>
      <ShieldCheck size={11} className="badge-icon" />
      <span>{level} Confidence</span>
    </span>
  );
}

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user';

  return (
    <div className={`message-row ${isUser ? 'message-row-user' : 'message-row-assistant'}`}>
      <div className={`message-bubble ${isUser ? 'bubble-user' : 'bubble-assistant'}`}>
        {!isUser && message.heading && (
          <h4 className="message-heading">{message.heading}</h4>
        )}

        <div className="message-body">
          {isUser ? (
            <p className="message-text">{message.text}</p>
          ) : (
            <div className="message-markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.text}
              </ReactMarkdown>
              {message.isGenerating && <span className="blinking-cursor">|</span>}
            </div>
          )}
        </div>

        {!isUser && message.bullets && message.bullets.length > 0 && (
          <ul className="message-bullets">
            {message.bullets.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ul>
        )}

        {/* Assistant Footer Info (Sources, Confidence, Time) */}
        {!isUser && (
          <div className="assistant-meta-strip">
            {message.sources && message.sources.length > 0 && (
              <div className="message-sources">
                <span className="sources-label">Sources:</span>
                <div className="source-tags-wrap">
                  {message.sources.map((s, i) => (
                    <span key={i} className="source-tag" title={s}>
                      <FileText size={10} className="source-icon" />
                      <span>{s}</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="message-footer-row">
              {message.confidence && (
                <div className="message-confidence">
                  <ConfidenceBadge level={message.confidence} />
                </div>
              )}
              {message.timestamp && (
                <span className="message-time">
                  <Clock size={10} className="time-icon" />
                  <span>{formatTime(message.timestamp)}</span>
                </span>
              )}
            </div>
          </div>
        )}

        {/* User Time */}
        {isUser && message.timestamp && (
          <span className="message-time user-time">
            {formatTime(message.timestamp)}
          </span>
        )}
      </div>
    </div>
  );
}
