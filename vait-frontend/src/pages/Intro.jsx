import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Intro.css';

const LETTERS = ['V', 'A', 'I', 'T'];

export default function Intro() {
  const navigate = useNavigate();

  /* Phase 1 — "Introducing VVIT's…" */
  const [phase1Visible, setPhase1Visible] = useState(false);
  const [phase1Out, setPhase1Out] = useState(false);

  /* Phase 2 — VAIT branding */
  const [visibleCount, setVisibleCount] = useState(0);
  const [showCaption, setShowCaption] = useState(false);

  /* Final exit */
  const [finalFadeOut, setFinalFadeOut] = useState(false);

  useEffect(() => {
    const t = [];

    /* ── PHASE 1 (0 – 6 s) ──────────────────────────────────────
       0.1 s  → fade-in begins (4 s ease-in-out)
       ~4 s   → fully visible
       5 s    → fade-out begins (0.8 s)
       ~5.8 s → blank screen                                      */
    t.push(setTimeout(() => setPhase1Visible(true), 100));
    t.push(setTimeout(() => setPhase1Out(true), 5000));

    /* ── PHASE 2 (6 – 14 s) ─────────────────────────────────────
       Letters appear one-by-one: 6 s, 7.25 s, 8.5 s, 9.75 s
       Caption fades in at 11 s
       Hold until 13.2 s → final fade-out (1 s)
       Navigate at 14.2 s                                         */
    LETTERS.forEach((_, i) => {
      t.push(setTimeout(() => setVisibleCount(i + 1), 6000 + i * 1250));
    });

    t.push(setTimeout(() => setShowCaption(true), 11000));
    t.push(setTimeout(() => setFinalFadeOut(true), 13200));
    t.push(setTimeout(() => navigate('/chat', { replace: true }), 14200));

    return () => t.forEach(clearTimeout);
  }, [navigate]);

  return (
    <div className={`intro-screen ${finalFadeOut ? 'intro-fade-out' : ''}`}>

      {/* Phase 1 — introductory text */}
      <div
        className={
          'intro-phase1' +
          (phase1Visible ? ' intro-phase1-visible' : '') +
          (phase1Out ? ' intro-phase1-out' : '')
        }
      >
        <p className="intro-phase1-text">
          Introducing VVIT&rsquo;s Artificial Intelligence Technology
        </p>
      </div>

      {/* Phase 2 — VAIT branding */}
      <div className="intro-phase2">
        <div className="intro-title" aria-label="VAIT">
          {LETTERS.map((letter, i) => (
            <span
              key={i}
              className={`intro-letter ${i < visibleCount ? 'intro-letter-visible' : ''}`}
            >
              {letter}
            </span>
          ))}
        </div>

        <p className={`intro-caption ${showCaption ? 'intro-caption-visible' : ''}`}>
          We Never Let You Wait for Anything
        </p>
      </div>
    </div>
  );
}
