import { Clock, AlertCircle, RotateCcw } from 'lucide-react';
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

export default function MessageBubble({ message, onRetry }) {
  const isUser = message.role === 'user';
  const hasText = Boolean(message.text && message.text.trim());
  const isThinking = !isUser && message.isGenerating && !hasText && !message.error;
  const isError = !isUser && (Boolean(message.error) || (!message.isGenerating && !hasText));
  const isStreaming = !isUser && message.isGenerating && hasText;

  return (
    <div className={`message-row ${isUser ? 'message-row-user' : 'message-row-assistant'}`}>
      <div
        className={`message-bubble ${isUser ? 'bubble-user' : 'bubble-assistant'} ${
          isThinking ? 'is-thinking' : ''
        } ${isStreaming ? 'is-streaming' : ''} ${isError ? 'is-error' : ''}`}
      >
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
            {isThinking ? (
              <div className="vait-thinking-state" aria-live="polite" aria-label="VAIT is thinking">
                <div className="vait-thinking-dots" aria-hidden="true">
                  <span className="vait-thinking-dot" />
                  <span className="vait-thinking-dot" />
                  <span className="vait-thinking-dot" />
                </div>
                <span className="vait-thinking-text">
                  {message.statusText || 'VAIT is thinking...'}
                </span>
              </div>
            ) : isError ? (
              <div className="vait-response-error" role="alert">
                <div className="vait-error-icon-wrap">
                  <AlertCircle size={15} className="vait-error-icon" />
                </div>
                <div className="vait-error-body">
                  <div className="vait-error-title">Unable to generate a response</div>
                  <div className="vait-error-desc">
                    {message.error || "I couldn't generate a response. Please try again."}
                  </div>
                  {onRetry && (
                    <button
                      type="button"
                      className="vait-retry-button"
                      onClick={() => onRetry(message)}
                      title="Retry generating response"
                    >
                      <RotateCcw size={12} />
                      <span>Retry</span>
                    </button>
                  )}
                </div>
              </div>
            ) : (
              <>
                {message.heading && (
                  <h4 className="message-heading">{message.heading}</h4>
                )}

                <UniversalResponseRenderer message={message} />

                {!message.isGenerating && message.timestamp && (
                  <div className="message-assistant-time-row">
                    <span className="message-time">
                      <Clock size={10} className="time-icon" />
                      <span>{formatTime(message.timestamp)}</span>
                    </span>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
