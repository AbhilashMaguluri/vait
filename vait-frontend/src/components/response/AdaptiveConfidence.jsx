import { ShieldCheck, AlertCircle } from 'lucide-react';

export default function AdaptiveConfidence({ level, isRefusal = false, showPill = false }) {
  if (!level) return null;

  // Only render if explicitly requested or if it's a refusal/low confidence situation where helpful
  if (level === 'High' && !showPill) return null;

  const classMap = {
    High: 'conf-high',
    Medium: 'conf-medium',
    Low: 'conf-low',
  };

  return (
    <div className={`vait-adaptive-confidence ${classMap[level] || ''}`}>
      {level === 'High' ? (
        <span className="conf-pill">
          <ShieldCheck size={11} />
          <span>High Authority</span>
        </span>
      ) : level === 'Medium' ? (
        <span className="conf-pill">
          <ShieldCheck size={11} />
          <span>Verified Context</span>
        </span>
      ) : isRefusal ? (
        <span className="conf-refusal-hint">
          <AlertCircle size={11} />
          <span>Official records do not explicitly state this information.</span>
        </span>
      ) : null}
    </div>
  );
}
