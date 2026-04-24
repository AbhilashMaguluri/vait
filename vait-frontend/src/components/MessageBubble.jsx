import './MessageBubble.css';

function formatTime(isoString) {
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
      {level} Confidence
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

        <p className="message-text">
          {message.text}
          {!isUser && message.isGenerating && <span className="blinking-cursor">|</span>}
        </p>

        {!isUser && message.bullets && message.bullets.length > 0 && (
          <ul className="message-bullets">
            {message.bullets.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ul>
        )}

        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            <span className="sources-label">Source:</span>
            {message.sources.map((s, i) => (
              <span key={i} className="source-tag">{s}</span>
            ))}
          </div>
        )}

        {!isUser && message.confidence && (
          <div className="message-confidence">
            <ConfidenceBadge level={message.confidence} />
          </div>
        )}

        <span className="message-time">{formatTime(message.timestamp)}</span>
      </div>
    </div>
  );
}
