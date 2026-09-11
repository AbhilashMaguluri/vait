import { ArrowDown, CheckCircle } from 'lucide-react';

export default function ProcessTimeline({ steps = [] }) {
  if (!steps || steps.length === 0) return null;

  return (
    <div className="vait-timeline-wrap">
      <div className="vait-timeline-list">
        {steps.map((step, idx) => (
          <div key={idx} className="vait-timeline-step">
            <div className="vait-step-indicator">
              <span className="vait-step-num">{idx + 1}</span>
              {idx < steps.length - 1 && <div className="vait-step-line" />}
            </div>
            <div className="vait-step-body">
              <h5 className="vait-step-title">{step.title}</h5>
              {step.description && <p className="vait-step-desc">{step.description}</p>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
