import { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import CodeBlockView from './CodeBlockView';
import FactHero from './FactHero';
import TransitionFlow from './TransitionFlow';
import AdaptiveSources from './AdaptiveSources';
import AdaptiveConfidence from './AdaptiveConfidence';
import CalloutBanner from './CalloutBanner';
import './UniversalResponse.css';

/**
 * Client-side auto-detector to ensure adaptive presentation
 * even on streamed responses, history items, or fallbacks.
 */
function detectFormat(text, explicitType) {
  if (explicitType && explicitType !== 'informational') {
    return explicitType;
  }

  const raw = text || '';

  if (/```(?:python|javascript|js|ts|c\+\+|cpp|c|java|sql|sh|bash|html|css)/i.test(raw)) {
    return 'code';
  }

  if (
    /Current Official Website/i.test(raw) ||
    /Legacy VVIT Website/i.test(raw) ||
    /\b(https:\/\/vvitu\.ac\.in\/|https:\/\/vvitguntur\.com\/)\b/.test(raw) && raw.length < 500
  ) {
    return 'factual';
  }

  if (
    /transition (?:from|to) vvit|institutional evolution|vvit university \(vvitu\)/i.test(raw) &&
    /1997|2022/i.test(raw)
  ) {
    return 'current_historical';
  }

  if (/\|\s*Feature|Aspect|Criteria\s*\|/i.test(raw) || /\|\s*CSE\s*\|\s*ECE\s*\|/i.test(raw)) {
    return 'comparison';
  }

  if (/\|\s*Category\s*\|\s*Amount\s*\|/i.test(raw) || /\|\s*Fee\s*\|/i.test(raw)) {
    return 'fee';
  }

  if (/\|\s*Event\s*\|\s*Date\s*\|/i.test(raw) || /\|\s*Semester\s*\|\s*Date\s*\|/i.test(raw)) {
    return 'schedule';
  }

  if (/-\s*\[\s*[\sxX]?\s*\]/i.test(raw) || /☐|☑/.test(raw)) {
    return 'checklist';
  }

  if (
    /###?\s*1\.\s*(?:Simple\s+)?Definition/i.test(raw) ||
    /##\s*1\.\s*Formal\s+Definition/i.test(raw)
  ) {
    return 'educational';
  }

  if (
    /^\s*1\.\s*\*\*.*\*\*/m.test(raw) &&
    /^\s*2\.\s*\*\*.*\*\*/m.test(raw) &&
    /^\s*3\.\s*\*\*.*\*\*/m.test(raw)
  ) {
    return 'process';
  }

  if (raw.length < 220 && !raw.includes('\n\n')) {
    return 'simple';
  }

  return 'informational';
}

/**
 * Extracts a candidate hero fact value and context from factual answers.
 */
function parseHeroFact(text) {
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean);
  let title = 'Official Record';
  let value = '';
  let url = '';
  let era = 'current';

  if (/Legacy VVIT Website/i.test(text)) {
    title = 'LEGACY VVIT INSTITUTE WEBSITE';
    value = 'vvitguntur.com';
    url = 'https://vvitguntur.com/';
    era = 'historical';
  } else if (/Current Official Website/i.test(text) || /vvitu\.ac\.in/i.test(text)) {
    title = 'CURRENT UNIVERSITY WEBSITE';
    value = 'vvitu.ac.in';
    url = 'https://vvitu.ac.in/';
    era = 'current';
  }

  return { title, value, url, era };
}

export default function UniversalResponseRenderer({ message }) {
  const text = message.text || '';
  const isGenerating = message.isGenerating;
  const explicitType = message.responseType || message.response_type;
  const sourceVisibility = message.source_visibility || message.sourceVisibility || 'none';

  const format = useMemo(() => detectFormat(text, explicitType), [text, explicitType]);
  const heroFact = useMemo(() => (format === 'factual' ? parseHeroFact(text) : null), [format, text]);

  // Markdown custom component overrides
  const markdownComponents = useMemo(
    () => ({
      code({ node, inline, className, children, ...props }) {
        const match = /language-(\w+)/.exec(className || '');
        if (!inline && (match || String(children).includes('\n'))) {
          return (
            <CodeBlockView
              language={match ? match[1] : 'text'}
              value={String(children).replace(/\n$/, '')}
            />
          );
        }
        return (
          <code className="vait-inline-code" {...props}>
            {children}
          </code>
        );
      },
      table({ children }) {
        return (
          <div className="vait-table-scroll-wrap">
            <table className="vait-adaptive-table">{children}</table>
          </div>
        );
      },
      blockquote({ children }) {
        return <CalloutBanner type="info">{children}</CalloutBanner>;
      },
    }),
    []
  );

  const isCompact = sourceVisibility === 'compact' || format === 'simple';
  const densityClass =
    text.length > 800 ? 'density-expanded' : text.length > 300 ? 'density-medium' : 'density-compact';

  return (
    <div className={`vait-universal-response format-${format} ${densityClass}`}>
      {/* 1. Fact Hero for Factual Lookups */}
      {format === 'factual' && heroFact && heroFact.value && (
        <FactHero
          title={heroFact.title}
          value={heroFact.value}
          url={heroFact.url}
          era={heroFact.era}
        />
      )}

      {/* 2. Institutional Evolution Flow for Transition Queries */}
      {format === 'current_historical' && <TransitionFlow />}

      {/* 3. Main Adaptive Markdown Body */}
      <div className="vait-response-body">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
          {text}
        </ReactMarkdown>
        {isGenerating && <span className="vait-blinking-cursor">|</span>}
      </div>

      {/* 4. Adaptive Sources & Confidence Footer */}
      {sourceVisibility !== 'none' && !isGenerating && (
        <div className="vait-response-footer">
          <AdaptiveSources
            sources={message.sources}
            structuredSources={message.structuredSources || message.structured_sources}
            isCompact={isCompact}
            sourceVisibility={sourceVisibility}
          />
          <AdaptiveConfidence level={message.confidence} isRefusal={message.is_refusal} />
        </div>
      )}
    </div>
  );
}
