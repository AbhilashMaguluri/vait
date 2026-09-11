import { Globe, FileText, ExternalLink, ShieldCheck, History } from 'lucide-react';

export default function AdaptiveSources({ sources = [], structuredSources = [], isCompact = false }) {
  // Normalize items
  const items = structuredSources && structuredSources.length > 0
    ? structuredSources
    : (sources || []).map((s) => ({
        title: typeof s === 'string' ? s : s?.title || 'Institutional Source',
        url: typeof s === 'string' && s.startsWith('http') ? s : s?.url || '',
        period_label: s?.period_label || (typeof s === 'string' && s.includes('vvitu.ac.in') ? 'VVITU Official Website — Current' : typeof s === 'string' && s.includes('vvitguntur.com') ? 'VVIT Legacy Website — Historical' : ''),
        type: s?.type || (typeof s === 'string' && s.startsWith('http') ? 'website' : 'document'),
      }));

  if (!items || items.length === 0) return null;

  // Compact display for simple/short answers
  if (isCompact || items.length === 1) {
    const first = items[0];
    const isCurrent = (first.period_label || '').toLowerCase().includes('current') || (first.url || '').includes('vvitu.ac.in');
    return (
      <div className="vait-sources-compact">
        <span className="vait-source-pill">
          {isCurrent ? <ShieldCheck size={11} className="pill-icon current-icon" /> : <History size={11} className="pill-icon historical-icon" />}
          <span className="pill-era">{isCurrent ? 'CURRENT' : 'HISTORICAL'}</span>
          <span className="pill-sep">·</span>
          {first.url ? (
            <a href={first.url} target="_blank" rel="noopener noreferrer" className="pill-link">
              <span>{first.title || first.url}</span>
              <ExternalLink size={9} />
            </a>
          ) : (
            <span className="pill-text">{first.title}</span>
          )}
        </span>
      </div>
    );
  }

  // Multi-source grid
  return (
    <div className="vait-sources-container">
      <div className="vait-sources-title-row">
        <span className="vait-sources-heading">Verified Sources</span>
      </div>
      <div className="vait-sources-grid">
        {items.map((item, idx) => {
          const isCurr = (item.period_label || '').toLowerCase().includes('current') || (item.url || '').includes('vvitu.ac.in');
          const isHist = (item.period_label || '').toLowerCase().includes('historical') || (item.url || '').includes('vvitguntur.com');
          return (
            <div key={idx} className={`vait-source-card ${isCurr ? 'source-card-current' : isHist ? 'source-card-historical' : ''}`}>
              <div className="vait-card-top">
                <span className={`vait-card-era-badge ${isCurr ? 'era-current' : isHist ? 'era-historical' : 'era-general'}`}>
                  {isCurr ? 'CURRENT · VVITU' : isHist ? 'HISTORICAL · VVIT' : 'DOCUMENT'}
                </span>
              </div>
              <div className="vait-card-title-row">
                {item.type === 'website' || item.url ? (
                  <Globe size={12} className="source-type-icon" />
                ) : (
                  <FileText size={12} className="source-type-icon" />
                )}
                {item.url ? (
                  <a href={item.url} target="_blank" rel="noopener noreferrer" className="source-card-link">
                    <span className="source-card-name" title={item.title}>{item.title}</span>
                    <ExternalLink size={10} className="ext-icon" />
                  </a>
                ) : (
                  <span className="source-card-name" title={item.title}>{item.title}</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
