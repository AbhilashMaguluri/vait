import { Routes, Route, Navigate } from 'react-router-dom';
import { ChatProvider } from './context/ChatContext';
import Intro from './pages/Intro';
import Chat from './pages/Chat';

export default function App() {
  return (
    <ChatProvider>
      <Routes>
        <Route path="/" element={<Intro />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </ChatProvider>
  );
}
