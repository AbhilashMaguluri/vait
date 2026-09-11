import { CheckSquare } from 'lucide-react';

export default function ChecklistGroup({ items = [], title = 'Required Items' }) {
  if (!items || items.length === 0) return null;

  return (
    <div className="vait-checklist-card">
      {title && (
        <div className="vait-checklist-header">
          <CheckSquare size={14} className="vait-checklist-icon" />
          <span className="vait-checklist-title">{title}</span>
        </div>
      )}
      <ul className="vait-checklist-list">
        {items.map((item, idx) => (
          <li key={idx} className="vait-checklist-item">
            <span className="vait-checkbox-box">
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <rect x="0.5" y="0.5" width="11" height="11" rx="2.5" stroke="currentColor" />
                <path d="M2.5 6L4.8 8.3L9.5 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
            <span className="vait-checklist-text">{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
