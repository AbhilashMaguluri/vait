import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import './Auth.css';

export default function AuthCallback() {
  const navigate = useNavigate();
  const { completeGoogleOAuth } = useAuth();
  const [error, setError] = useState('');

  useEffect(() => {
    let alive = true;

    async function finishOAuth() {
      const params = new URLSearchParams(window.location.hash.replace(/^#/, ''));
      const queryParams = new URLSearchParams(window.location.search);
      const oauthError = params.get('error') || queryParams.get('error');
      const token = params.get('access_token');

      try {
        if (oauthError) {
          throw new Error(oauthError);
        }
        await completeGoogleOAuth(token);
        if (alive) navigate('/chat', { replace: true });
      } catch (err) {
        if (alive) setError(err?.message || 'Google sign-in failed.');
      }
    }

    finishOAuth();
    return () => {
      alive = false;
    };
  }, [completeGoogleOAuth, navigate]);

  return (
    <main className="auth-page">
      <section className="auth-brand-panel" aria-label="VAIT identity">
        <Link className="auth-brand" to="/login">VAIT</Link>
        <div className="auth-brand-copy">
          <span className="auth-eyebrow">Google OAuth</span>
          <h1>Completing secure sign-in</h1>
          <p>Your VAIT session is being prepared.</p>
        </div>
        <div className="auth-status-strip">
          <span>Verified sources</span>
          <span>Institutional RAG</span>
          <span>Secure session</span>
        </div>
      </section>

      <section className="auth-form-panel" aria-label="OAuth callback status">
        <div className="auth-heading">
          <div>
            <h2>{error ? 'Sign-in needs attention' : 'Signing you in'}</h2>
            <p>{error || 'Please wait while Google returns you to VAIT.'}</p>
          </div>
        </div>
        {error ? (
          <Link className="auth-primary-btn" to="/login">
            Back to login
          </Link>
        ) : (
          <div className="auth-loading-screen auth-loading-inline">
            <span className="auth-loading-mark">VAIT</span>
          </div>
        )}
      </section>
    </main>
  );
}
