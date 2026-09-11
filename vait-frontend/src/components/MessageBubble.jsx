import { Clock } from 'lucide-react';
import UniversalResponseRenderer from './response/UniversalResponseRenderer';
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

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user';

  return (
    <div className={`message-row ${isUser ? 'message-row-user' : 'message-row-assistant'}`}>
      <div className={`message-bubble ${isUser ? 'bubble-user' : 'bubble-assistant'}`}>
        {isUser ? (
          <>
            <div className="message-body">
              <p className="message-text">{message.text}</p>
            </div>
            {message.timestamp && (
              <span className="message-time user-time">
                {formatTime(message.timestamp)}
              </span>
            )}
          </>
        ) : (
          <>
            {message.heading && (
              <h4 className="message-heading">{message.heading}</h4>
            )}

            <UniversalResponseRenderer message={message} />

            {message.timestamp && (
              <div className="message-assistant-time-row">
                <span className="message-time">
                  <Clock size={10} className="time-icon" />
                  <span>{formatTime(message.timestamp)}</span>
                </span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
