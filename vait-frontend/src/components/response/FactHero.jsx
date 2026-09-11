import { Globe, Building, Award, MapPin, CheckCircle2, Shield } from 'lucide-react';

export default function FactHero({ title, value, era, contextText, url }) {
  const isCurrent = era !== 'historical';

  return (
    <div className={`vait-fact-hero ${isCurrent ? 'vait-fact-current' : 'vait-fact-historical'}`}>
      <div className="vait-fact-top">
        <span className="vait-fact-badge">
          {isCurrent ? (
            <>
              <Shield size={11} className="vait-badge-icon" />
              <span>CURRENT · Official VVITU Source</span>
            </>
          ) : (
            <>
              <Building size={11} className="vait-badge-icon" />
              <span>HISTORICAL · VVIT Legacy Source</span>
            </>
          )}
        </span>
        {title && <span className="vait-fact-label">{title}</span>}
      </div>

      <div className="vait-fact-main">
        {url ? (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="vait-fact-value vait-fact-link"
          >
            <Globe size={18} className="vait-fact-link-icon" />
            <span>{value || url}</span>
          </a>
        ) : (
          <div className="vait-fact-value">{value}</div>
        )}
      </div>

      {contextText && (
        <div className="vait-fact-context">
          <p>{contextText}</p>
        </div>
      )}
    </div>
  );
}
