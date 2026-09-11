import { BookOpen, Lightbulb, Cog, Sparkles, CheckCircle2 } from 'lucide-react';

export default function EducationalSections({ sections = [], keyTakeaways = [] }) {
  const stageIcons = [Lightbulb, BookOpen, Cog, Sparkles, CheckCircle2];

  return (
    <div className="vait-educational-wrap">
      {sections.map((sec, idx) => {
        const Icon = stageIcons[idx % stageIcons.length];
        return (
          <div key={idx} className="vait-edu-section">
            <div className="vait-edu-header">
              <span className="vait-edu-pill">
                <span className="vait-edu-num">{String(idx + 1).padStart(2, '0')}</span>
                <span className="vait-edu-title">{sec.title}</span>
              </span>
            </div>
            <div className="vait-edu-body">
              <p>{sec.content}</p>
            </div>
          </div>
        );
      })}

      {keyTakeaways && keyTakeaways.length > 0 && (
        <div className="vait-takeaways-card">
          <div className="vait-takeaways-header">
            <CheckCircle2 size={15} className="vait-takeaway-icon" />
            <span>Key Takeaways & Exam Summary</span>
          </div>
          <ul className="vait-takeaways-list">
            {keyTakeaways.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
