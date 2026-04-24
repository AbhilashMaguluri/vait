import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { buildApiUrl } from '../utils/apiConfig';

const AuthContext = createContext(null);
const TOKEN_KEY = 'vait.accessToken';
const USER_KEY = 'vait.user';

function readStoredUser() {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    localStorage.removeItem(USER_KEY);
    return null;
  }
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function apiErrorMessage(detail, fallback) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const message = detail
      .map((item) => item?.msg || item?.message || '')
      .filter(Boolean)
      .join(' ');
    return message || fallback;
  }
  return detail?.message || fallback;
}

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY) || '');
  const [user, setUser] = useState(readStoredUser);
  const [initializing, setInitializing] = useState(() => Boolean(localStorage.getItem(TOKEN_KEY) && !readStoredUser()));

  const clearSession = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    setToken('');
    setUser(null);
  }, []);

  const storeSession = useCallback((payload) => {
    const nextToken = payload?.access_token || '';
    const nextUser = payload?.user || null;

    if (!nextToken || !nextUser) {
      throw new Error('Invalid authentication response.');
    }

    localStorage.setItem(TOKEN_KEY, nextToken);
    localStorage.setItem(USER_KEY, JSON.stringify(nextUser));
    setToken(nextToken);
    setUser(nextUser);
    return nextUser;
  }, []);

  const requestAuth = useCallback(async (path, body) => {
    const response = await fetch(buildApiUrl(path), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await readJson(response);

    if (!response.ok) {
      throw new Error(apiErrorMessage(data?.detail, 'Authentication failed.'));
    }

    return storeSession(data);
  }, [storeSession]);

  const login = useCallback(
    ({ identifier, password }) => requestAuth('/api/vait/auth/login', { identifier, password }),
    [requestAuth]
  );

  const signup = useCallback(
    ({ fullName, email, password, username }) =>
      requestAuth('/api/vait/auth/signup', {
        full_name: fullName,
        email,
        password,
        username: username || undefined,
      }),
    [requestAuth]
  );

  const loadMe = useCallback(async (activeToken = token) => {
    if (!activeToken) return null;

    const response = await fetch(buildApiUrl('/api/vait/auth/me'), {
      headers: { Authorization: `Bearer ${activeToken}` },
    });
    const data = await readJson(response);

    if (!response.ok) {
      clearSession();
      throw new Error(apiErrorMessage(data?.detail, 'Session expired.'));
    }

    localStorage.setItem(USER_KEY, JSON.stringify(data.user));
    setUser(data.user);
    return data.user;
  }, [clearSession, token]);

  const completeGoogleOAuth = useCallback(async (oauthToken) => {
    if (!oauthToken) {
      throw new Error('Missing Google session token.');
    }
    localStorage.setItem(TOKEN_KEY, oauthToken);
    setToken(oauthToken);
    return loadMe(oauthToken);
  }, [loadMe]);

  const continueWithGoogle = useCallback(() => {
    const callbackUrl = `${window.location.origin}/auth/callback`;
    const url = `${buildApiUrl('/api/vait/auth/google/login')}?next=${encodeURIComponent(callbackUrl)}`;
    window.location.assign(url);
  }, []);

  const logout = useCallback(async () => {
    const activeToken = token;
    clearSession();

    if (!activeToken) return;
    try {
      await fetch(buildApiUrl('/api/vait/auth/logout'), {
        method: 'POST',
        headers: { Authorization: `Bearer ${activeToken}` },
      });
    } catch {
      // Local logout should always complete even if the API is unreachable.
    }
  }, [clearSession, token]);

  useEffect(() => {
    let alive = true;

    async function hydrateSession() {
      if (!token) {
        setInitializing(false);
        return;
      }

      if (user) {
        setInitializing(false);
        return;
      }

      try {
        await loadMe(token);
      } catch {
        // loadMe clears invalid sessions.
      } finally {
        if (alive) setInitializing(false);
      }
    }

    hydrateSession();
    return () => {
      alive = false;
    };
  }, [loadMe, token, user]);

  const value = useMemo(
    () => ({
      token,
      user,
      initializing,
      isAuthenticated: Boolean(token),
      login,
      signup,
      logout,
      continueWithGoogle,
      completeGoogleOAuth,
    }),
    [completeGoogleOAuth, continueWithGoogle, initializing, login, logout, signup, token, user]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}

export function ProtectedRoute({ children }) {
  const { initializing, isAuthenticated } = useAuth();
  const location = useLocation();

  if (initializing) {
    return (
      <div className="auth-loading-screen">
        <span className="auth-loading-mark">VAIT</span>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  return children;
}
