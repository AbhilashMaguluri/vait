import { ArrowRight, Landmark, GraduationCap, Clock } from 'lucide-react';

export default function TransitionFlow() {
  return (
    <div className="vait-transition-card">
      <div className="vait-transition-header">
        <Landmark size={14} className="vait-transition-icon" />
        <span className="vait-transition-title">Institutional Evolution Timeline</span>
      </div>

      <div className="vait-transition-stages">
        {/* Stage 1: Legacy Institute */}
        <div className="vait-stage vait-stage-legacy">
          <div className="vait-stage-badge">1997 – 2022</div>
          <h4 className="vait-stage-title">VVIT</h4>
          <p className="vait-stage-subtitle">Vasireddy Venkatadri Institute of Technology</p>
          <div className="vait-stage-desc">
            Affiliated engineering college under JNTU Kakinada with autonomous status and NAAC 'A' grade.
          </div>
          <div className="vait-stage-portal">
            <span className="portal-tag">Archival Portal:</span>
            <a href="https://vvitguntur.com/" target="_blank" rel="noopener noreferrer">vvitguntur.com</a>
          </div>
        </div>

        {/* Transition Arrow */}
        <div className="vait-transition-arrow-wrap">
          <div className="vait-arrow-line" />
          <div className="vait-arrow-pill">
            <Clock size={11} />
            <span>State University Autonomy</span>
          </div>
          <ArrowRight size={16} className="vait-arrow-glyph" />
        </div>

        {/* Stage 2: Current University */}
        <div className="vait-stage vait-stage-current">
          <div className="vait-stage-badge current-badge">2022 – Present</div>
          <h4 className="vait-stage-title">VVITU</h4>
          <p className="vait-stage-subtitle">VVIT University</p>
          <div className="vait-stage-desc">
            Full-fledged multidisciplinary university with broadened degree-granting autonomy under UGC.
          </div>
          <div className="vait-stage-portal">
            <span className="portal-tag current-tag">Official University Portal:</span>
            <a href="https://vvitu.ac.in/" target="_blank" rel="noopener noreferrer">vvitu.ac.in</a>
          </div>
        </div>
      </div>
    </div>
  );
}
