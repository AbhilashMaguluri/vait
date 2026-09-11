import { useState, useCallback } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check, Terminal } from 'lucide-react';

export default function CodeBlockView({ language, value }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    if (!value) return;
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  }, [value]);

  const langLabel = (language || 'code').toUpperCase();

  return (
    <div className="vait-code-window">
      <div className="vait-code-header">
        <div className="vait-code-title">
          <Terminal size={13} className="vait-code-icon" />
          <span className="vait-code-lang">{langLabel}</span>
        </div>
        <button
          type="button"
          onClick={handleCopy}
          className="vait-code-copy-btn"
          aria-label="Copy code to clipboard"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check size={12} className="vait-copy-check" />
              <span>Copied!</span>
            </>
          ) : (
            <>
              <Copy size={12} />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <div className="vait-code-content">
        <SyntaxHighlighter
          language={language || 'text'}
          style={oneDark}
          customStyle={{
            margin: 0,
            padding: '14px 16px',
            fontSize: '12.5px',
            lineHeight: '1.6',
            background: '#161822',
            borderRadius: '0 0 8px 8px',
          }}
          wrapLongLines={true}
        >
          {value}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
