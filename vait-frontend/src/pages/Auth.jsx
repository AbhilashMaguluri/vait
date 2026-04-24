import { useEffect, useMemo, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Lock, Mail, ShieldCheck, UserRound } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import './Auth.css';

const initialForm = {
  fullName: '',
  username: '',
  identifier: '',
  email: '',
  password: '',
};

export default function Auth({ mode = 'login' }) {
  const isSignup = mode === 'signup';
  const navigate = useNavigate();
  const location = useLocation();
  const { login, signup, continueWithGoogle, isAuthenticated } = useAuth();
  const [form, setForm] = useState(initialForm);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const destination = useMemo(
    () => location.state?.from?.pathname || '/chat',
    [location.state]
  );

  useEffect(() => {
    setError('');
    setSubmitting(false);
  }, [mode]);

  useEffect(() => {
    if (isAuthenticated) {
      navigate(destination, { replace: true });
    }
  }, [destination, isAuthenticated, navigate]);

  const updateField = (field) => (event) => {
    setForm((prev) => ({ ...prev, [field]: event.target.value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setSubmitting(true);

    try {
      if (isSignup) {
        await signup({
          fullName: form.fullName.trim(),
          username: form.username.trim(),
          email: form.email.trim(),
          password: form.password,
        });
      } else {
        await login({
          identifier: form.identifier.trim(),
          password: form.password,
        });
      }
      navigate(destination, { replace: true });
    } catch (err) {
      setError(err?.message || 'Authentication failed.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="auth-page">
      <section className="auth-brand-panel" aria-label="VAIT identity">
        <Link className="auth-brand" to="/login" aria-label="VAIT login">
          VAIT
        </Link>
        <div className="auth-brand-copy">
          <span className="auth-eyebrow">Official Academic Mode</span>
          <h1>VVIT's Artificial Intelligence Technology</h1>
          <p>Secure access for students, trainers, and administrators.</p>
        </div>
        <div className="auth-status-strip">
          <span>Verified sources</span>
          <span>Institutional RAG</span>
          <span>Google OAuth ready</span>
        </div>
      </section>

      <section className="auth-form-panel" aria-label={isSignup ? 'Create account' : 'Sign in'}>
        <div className="auth-tabs" role="tablist" aria-label="Authentication mode">
          <Link className={`auth-tab ${!isSignup ? 'auth-tab-active' : ''}`} to="/login">
            Login
          </Link>
          <Link className={`auth-tab ${isSignup ? 'auth-tab-active' : ''}`} to="/signup">
            Sign up
          </Link>
        </div>

        <div className="auth-heading">
          <ShieldCheck size={18} aria-hidden="true" />
          <div>
            <h2>{isSignup ? 'Create your VAIT account' : 'Welcome back'}</h2>
            <p>{isSignup ? 'Use your college email to get started.' : 'Sign in to continue to VAIT.'}</p>
          </div>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          {isSignup ? (
            <>
              <label className="auth-field">
                <span>Full name</span>
                <div className="auth-input-wrap">
                  <UserRound size={17} aria-hidden="true" />
                  <input
                    type="text"
                    value={form.fullName}
                    onChange={updateField('fullName')}
                    autoComplete="name"
                    required
                    minLength={2}
                  />
                </div>
              </label>

              <label className="auth-field">
                <span>Username</span>
                <div className="auth-input-wrap">
                  <UserRound size={17} aria-hidden="true" />
                  <input
                    type="text"
                    value={form.username}
                    onChange={updateField('username')}
                    autoComplete="username"
                    minLength={3}
                    placeholder="Optional"
                  />
                </div>
              </label>

              <label className="auth-field">
                <span>Email</span>
                <div className="auth-input-wrap">
                  <Mail size={17} aria-hidden="true" />
                  <input
                    type="email"
                    value={form.email}
                    onChange={updateField('email')}
                    autoComplete="email"
                    required
                  />
                </div>
              </label>
            </>
          ) : (
            <label className="auth-field">
              <span>Email or username</span>
              <div className="auth-input-wrap">
                <Mail size={17} aria-hidden="true" />
                <input
                  type="text"
                  value={form.identifier}
                  onChange={updateField('identifier')}
                  autoComplete="username"
                  required
                  minLength={3}
                />
              </div>
            </label>
          )}

          <label className="auth-field">
            <span>Password</span>
            <div className="auth-input-wrap">
              <Lock size={17} aria-hidden="true" />
              <input
                type="password"
                value={form.password}
                onChange={updateField('password')}
                autoComplete={isSignup ? 'new-password' : 'current-password'}
                required
                minLength={isSignup ? 6 : 1}
              />
            </div>
          </label>

          {error && <p className="auth-error">{error}</p>}

          <button className="auth-primary-btn" type="submit" disabled={submitting}>
            {submitting ? 'Please wait...' : isSignup ? 'Create account' : 'Login'}
          </button>
        </form>

        <div className="auth-divider">
          <span />
          <p>or</p>
          <span />
        </div>

        <button className="auth-google-btn" type="button" onClick={continueWithGoogle}>
          <span className="auth-google-mark" aria-hidden="true">G</span>
          Continue with Google
        </button>
      </section>
    </main>
  );
}
