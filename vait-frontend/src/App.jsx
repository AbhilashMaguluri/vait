import { Component } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { ChatProvider } from './context/ChatContext';
import Intro from './pages/Intro';
import Chat from './pages/Chat';

class AppErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, message: '' };
  }

  static getDerivedStateFromError(error) {
    return {
      hasError: true,
      message: error?.message || 'Unknown runtime error',
    };
  }

  componentDidCatch(error, errorInfo) {
    console.error('[VAIT] Frontend runtime crash:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            minHeight: '100vh',
            display: 'grid',
            placeItems: 'center',
            padding: '24px',
            background: '#f5f5f3',
            color: '#2c2c3a',
            fontFamily: 'Inter, sans-serif',
          }}
        >
          <div style={{ maxWidth: '720px', textAlign: 'left' }}>
            <h1 style={{ marginBottom: '12px', fontSize: '1.5rem' }}>
              VAIT frontend encountered an error
            </h1>
            <p style={{ marginBottom: '8px' }}>
              The application failed to render. Check browser console logs for details.
            </p>
            <pre
              style={{
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                background: '#ffffff',
                border: '1px solid #d8d8de',
                borderRadius: '8px',
                padding: '12px',
              }}
            >
              {this.state.message}
            </pre>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default function App() {
  return (
    <AppErrorBoundary>
      <ChatProvider>
        <Routes>
          <Route path="/" element={<Intro />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </ChatProvider>
    </AppErrorBoundary>
  );
}
