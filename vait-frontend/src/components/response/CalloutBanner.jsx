import { AlertCircle, Info, AlertTriangle, CheckCircle } from 'lucide-react';

export default function CalloutBanner({ type = 'info', title, children }) {
  const configs = {
    warning: {
      icon: AlertTriangle,
      className: 'vait-callout-warning',
      defaultTitle: 'Important Advisory',
    },
    danger: {
      icon: AlertCircle,
      className: 'vait-callout-danger',
      defaultTitle: 'Notice',
    },
    success: {
      icon: CheckCircle,
      className: 'vait-callout-success',
      defaultTitle: 'Verified Record',
    },
    info: {
      icon: Info,
      className: 'vait-callout-info',
      defaultTitle: 'Institutional Note',
    },
  };

  const current = configs[type] || configs.info;
  const Icon = current.icon;

  return (
    <div className={`vait-callout-banner ${current.className}`}>
      <div className="vait-callout-header">
        <Icon size={14} className="vait-callout-icon" />
        <span className="vait-callout-title">{title || current.defaultTitle}</span>
      </div>
      <div className="vait-callout-body">{children}</div>
    </div>
  );
}
